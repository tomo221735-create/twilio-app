# server.py
# 完成版 Flask サーバ（予約 + SMS（キャンセルURL付き）統合）
# 元のコードをベースに、CANCEL_BASE_URL と build_sms_text を追加し、
# 予約確定時の SMS にキャンセル URL を含めるように修正しています。

import os
import json
import tempfile
import threading
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple

from flask import Flask, request, jsonify, Response
import requests
from requests.auth import HTTPBasicAuth

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- 環境変数 ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
GAS_URL = os.environ.get("GAS_URL")               # スプレッドシート書き込み用（必須ではない）
GAS_CHECK_URL = os.environ.get("GAS_CHECK_URL")   # 予約チェック用（任意）
CHECK_RESERVE_TOKEN = os.environ.get("CHECK_RESERVE_TOKEN", "")
SEND_REPEAT_TO_GAS = os.environ.get("SEND_REPEAT_TO_GAS", "false").lower() in ("1", "true", "yes")

# 追加: キャンセル用ベースURL（例: https://example.com/cancel）
CANCEL_BASE_URL = os.environ.get("CANCEL_BASE_URL", "")

OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

# Twilio 送信用の発信元番号（E.164 形式の文字列にしてください）
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")

# -----------------------
# Helpers: TwiML response
# -----------------------
def twiml_say(text: str, voice: str = "woman", language: str = "ja-JP") -> Response:
    tw = f'<?xml version="1.0" encoding="UTF-8"?><Response><Say voice="{voice}" language="{language}">{escape_xml(text)}</Say></Response>'
    return Response(tw, mimetype="text/xml")

def escape_xml(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# -----------------------
# 録音ダウンロード
# -----------------------
def download_recording(recording_url: str, timeout: int = 15) -> Optional[str]:
    try:
        url = recording_url
        if not url.lower().endswith((".mp3", ".wav", ".ogg")):
            url = recording_url + ".mp3"
        auth = None
        if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
            auth = HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        logger.info("録音ダウンロード: %s", url)
        r = requests.get(url, auth=auth, timeout=timeout)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(r.content)
        tmp.flush()
        tmp.close()
        logger.info("録音を一時保存: %s", tmp.name)
        return tmp.name
    except Exception:
        logger.exception("録音ダウンロード失敗")
        return None

# -----------------------
# 文字起こし（OpenAI）
# -----------------------
def transcribe_via_http(file_path: str, model: str = "gpt-4o-mini-transcribe") -> Optional[str]:
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY が未設定です")
        return None
    try:
        with open(file_path, "rb") as f:
            files = {"file": ("speech.mp3", f, "audio/mpeg")}
            data = {"model": model}
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            resp = requests.post(OPENAI_TRANSCRIBE_URL, headers=headers, data=data, files=files, timeout=60)
            j = resp.json()
            logger.info("transcription raw json: %s", json.dumps(j, ensure_ascii=False))
            text = None
            if isinstance(j, dict):
                text = j.get("text") or j.get("transcript") or j.get("output_text")
                if not text and "data" in j and isinstance(j["data"], list) and j["data"]:
                    first = j["data"][0]
                    if isinstance(first, dict):
                        text = first.get("text")
                if not text and "choices" in j and isinstance(j["choices"], list) and j["choices"]:
                    c0 = j["choices"][0]
                    if isinstance(c0, dict):
                        text = c0.get("text")
            logger.info("抽出された文字起こし: %s", text)
            return text
    except Exception:
        logger.exception("文字起こしエラー（HTTP）")
        return None

# -----------------------
# OpenAI Responses 呼び出し（汎用）
# -----------------------
def call_ai_via_http(prompt: str, model: str = "gpt-4o-mini", timeout: int = 30) -> Optional[str]:
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY が未設定です")
        return None
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": model, "input": prompt}
        resp = requests.post(OPENAI_RESPONSES_URL, headers=headers, json=payload, timeout=timeout)
        j = resp.json()
        logger.info("responses raw json: %s", json.dumps(j, ensure_ascii=False))
        out = None
        if isinstance(j, dict):
            out = j.get("output_text")
            if not out and "choices" in j and isinstance(j["choices"], list) and j["choices"]:
                c0 = j["choices"][0]
                if isinstance(c0, dict):
                    out = c0.get("text")
            if not out and "output" in j and isinstance(j["output"], list) and j["output"]:
                first = j["output"][0]
                if isinstance(first, dict):
                    content = first.get("content")
                    if isinstance(content, list) and content:
                        c0 = content[0]
                        if isinstance(c0, dict):
                            out = c0.get("text")
        logger.info("AI 抽出結果: %s", out)
        return out
    except Exception:
        logger.exception("AI 呼び出しエラー（HTTP）")
        return None

# -----------------------
# GAS 送信（スプレッドシート反映）
# -----------------------
def send_to_gas(data: dict, retries: int = 1) -> Optional[requests.Response]:
    if not GAS_URL:
        logger.info("GAS_URL 未設定のため送信スキップ")
        return None
    try:
        logger.info("GAS送信開始: %s", data)
        headers = {"Content-Type": "application/json; charset=utf-8"}
        resp = requests.post(GAS_URL, json=data, headers=headers, timeout=10)
        logger.info("GAS HTTP status: %s", resp.status_code)
        logger.info("GAS response text: %s", resp.text)
        if resp.status_code >= 400 and retries > 0:
            time.sleep(1)
            return send_to_gas(data, retries=retries-1)
        return resp
    except requests.exceptions.RequestException:
        logger.exception("GAS送信例外")
        if retries > 0:
            time.sleep(1)
            return send_to_gas(data, retries=retries-1)
        return None

# -----------------------
# 名前抽出（漢字推測 + 読み）
# -----------------------
def extract_name_and_reading(text: str) -> Tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    prompt = f"""次の発言から、もし人名が含まれていれば必ずJSONで返してください。
JSONのキーは name_kanji と reading の2つのみとし、reading はひらがなで返してください。
name_kanji はAIの推測による漢字表記（確定ではない）として返してください。
名前が無ければ {{"name_kanji": null, "reading": null}} を返してください。
発言: 「{text}」
出力例: {{"name_kanji":"山田太郎","reading":"やまだたろう"}}"""
    res = call_ai_via_http(prompt)
    if not res:
        return None, None
    try:
        start = res.find("{")
        end = res.rfind("}") + 1
        jstr = res[start:end]
        j = json.loads(jstr)
        return j.get("name_kanji"), j.get("reading")
    except Exception:
        logger.exception("名前抽出パース失敗")
        return None, None

# -----------------------
# 日時と名前抽出（プロンプトに今日の日付を渡す）
# -----------------------
def extract_datetime_and_name(text: str, today: Optional[datetime] = None) -> Optional[dict]:
    if not text:
        return None
    if today is None:
        today = datetime.now()
    today_iso = today.date().isoformat()
    prompt = f"""次の発言から希望日時と名前（読み）を抽出して、必ずJSONで返してください。
キーは requested_date (YYYY-MM-DD), requested_time (HH:MM), name_reading (ひらがな), name_kanji (漢字推測 or null) の4つのみとしてください。
今日の日付は {today_iso} です。'明日' や '来週' といった相対表現は今日の日付を基準に解釈してください。
発言: 「{text}」
出力例: {{"requested_date":"2026-03-26","requested_time":"15:00","name_reading":"やまだたろう","name_kanji":"山田太郎"}}
日時や名前が不明瞭な場合は null を入れてください。"""
    res = call_ai_via_http(prompt)
    if not res:
        return None
    try:
        start = res.find("{")
        end = res.rfind("}") + 1
        jstr = res[start:end]
        j = json.loads(jstr)
        logger.info("日時抽出結果: %s", j)
        return j
    except Exception:
        logger.exception("日時抽出パース失敗")
        return None

# -----------------------
# GAS checkReserve 呼び出し（原子的に予約）
# -----------------------
def try_reserve(date: str, timeStart: str, name_reading: str, name_kanji: str, caller: str, notes: str = "") -> dict:
    if not GAS_CHECK_URL:
        logger.error("GAS_CHECK_URL が未設定です")
        return {"status": "error", "message": "no_gas_check_url"}
    payload = {
        "date": date,
        "timeStart": timeStart,
        "name_reading": name_reading,
        "name_kanji": name_kanji,
        "caller": caller,
        "notes": notes,
        "token": CHECK_RESERVE_TOKEN or ""
    }
    try:
        resp = requests.post(GAS_CHECK_URL, json=payload, timeout=10)
        logger.info("GAS checkReserve status: %s", resp.status_code)
        logger.info("GAS checkReserve raw text: %s", resp.text)
        try:
            j = resp.json()
            return j
        except ValueError:
            logger.warning("GAS checkReserve returned non-JSON response")
            text = (resp.text or "").strip()
            if text.upper() == "OK":
                return {"status": "ok", "written": "unknown", "raw_text": text}
            return {"status": "error", "message": "invalid_response", "text": text}
    except requests.exceptions.RequestException:
        logger.exception("GAS checkReserve 呼び出し失敗")
        return {"status": "error", "message": "request_failed"}

# -----------------------
# 日付/時刻検証と補正
# -----------------------
def validate_and_normalize_datetime(parsed: dict, original_text: str) -> Tuple[Optional[str], Optional[str]]:
    rd = parsed.get("requested_date") if parsed else None
    rt = parsed.get("requested_time") if parsed else None
    try:
        if rd:
            d = datetime.strptime(rd, "%Y-%m-%d").date()
            if d < datetime.now().date():
                if "明日" in original_text or "あした" in original_text:
                    newd = (datetime.now().date() + timedelta(days=1)).isoformat()
                    logger.info("過去日付を補正: %s -> %s", rd, newd)
                    rd = newd
                else:
                    logger.warning("解析された日付が過去です: %s", rd)
    except Exception:
        logger.exception("日付パース失敗")
        rd = None
    try:
        if rt:
            datetime.strptime(rt, "%H:%M")
    except Exception:
        logger.exception("時刻パース失敗")
        rt = None
    return rd, rt

# -----------------------
# Twilio SMS 送信ユーティリティ
# -----------------------
def send_sms(to_number: str, body: str) -> bool:
    """
    Twilio REST API を使って SMS を送信します。
    環境変数 TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_PHONE_NUMBER が必要です。
    失敗時はログに出力して False を返します。
    """
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER):
        logger.warning("Twilio 設定が不完全なため SMS を送信しません。to=%s body=%s", to_number, body)
        return False
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
        data = {
            "From": TWILIO_PHONE_NUMBER,
            "To": to_number,
            "Body": body
        }
        auth = HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        resp = requests.post(url, data=data, auth=auth, timeout=10)
        logger.info("Twilio status: %s text: %s", resp.status_code, resp.text)
        if resp.status_code >= 400:
            logger.error("Twilio SMS 送信失敗: %s", resp.text)
            return False
        return True
    except Exception:
        logger.exception("Twilio SMS 送信例外")
        return False

# -----------------------
# SMS 文面生成（キャンセルURL付き）
# -----------------------
def build_sms_text(date: str, timeStart: str, reservation_id: str) -> str:
    try:
        d = datetime.strptime(date, "%Y-%m-%d")
        weekday = ["月","火","水","木","金","土","日"][d.weekday()]

        # CANCEL_BASE_URL が設定されていればフル URL を作る
        if CANCEL_BASE_URL:
            # 既にクエリがあるかどうかを簡易判定して & または ? を付与
            sep = "&" if "?" in CANCEL_BASE_URL else "?"
            cancel_url = f"{CANCEL_BASE_URL}{sep}rid={reservation_id}"
        else:
            cancel_url = f"?rid={reservation_id}"

        return (
            f"{date}（{weekday}）の{timeStart}にご予約を承りました。\n"
            f"キャンセルはこちら：{cancel_url}"
        )
    except Exception:
        # フォールバック文面
        if CANCEL_BASE_URL:
            sep = "&" if "?" in CANCEL_BASE_URL else "?"
            return f"{date} {timeStart} のご予約です。\nキャンセル：{CANCEL_BASE_URL}{sep}rid={reservation_id}"
        return f"{date} {timeStart} のご予約です。キャンセルID: {reservation_id}"

# -----------------------
# バックグラウンド repeat 保存処理
# -----------------------
def background_repeat_process(user_text: Optional[str], caller: str = ""):
    try:
        if not user_text:
            if SEND_REPEAT_TO_GAS:
                send_to_gas({"raw": None, "parsed": {"step": "repeat", "error": "no_transcription"}, "target": "audio", "token": CHECK_RESERVE_TOKEN})
            return
        name_kanji, reading = extract_name_and_reading(user_text)
        prompt = f"""あなたは実在するクリニックの受付スタッフです。
電話応対として自然で聞き取りやすい日本語で話してください。
ユーザーの発言を一文で自然に復唱してください（案内はしない）。
ユーザーの発言:
{user_text}
"""
        reply = call_ai_via_http(prompt)
        if not reply:
            reply = f"{user_text}ですね。"
        parsed = {"step": "repeat", "reply": reply}
        if name_kanji:
            parsed["name_kanji"] = name_kanji
        if reading:
            parsed["reading"] = reading
        payload = {"raw": user_text, "parsed": parsed, "caller": caller, "target": "audio", "token": CHECK_RESERVE_TOKEN}
        if SEND_REPEAT_TO_GAS:
            send_to_gas(payload)
        else:
            logger.info("repeat processed (not sent to GAS): %s", parsed)
    except Exception:
        logger.exception("バックグラウンド repeat エラー")

# -----------------------
# バックグラウンド予約処理（録音URLとcallerを受け取る）
# -----------------------
def process_reservation_background(recording_url: str, caller: str):
    try:
        audio_file = download_recording(recording_url)
        user_text = None
        if audio_file:
            user_text = transcribe_via_http(audio_file)
            try:
                os.remove(audio_file)
            except Exception:
                pass

        # まず必ず audio に生ログを送る（音声本文・解析結果のログ）
        if user_text:
            parsed = extract_datetime_and_name(user_text, today=datetime.now())
            send_to_gas({
                "raw": user_text,
                "parsed": {"step": "reservation", "parsed": parsed},
                "caller": caller,
                "target": "audio",
                "token": CHECK_RESERVE_TOKEN
            })
        else:
            # 文字起こし失敗ログを audio に残す（録音URLをメモとして残す）
            send_to_gas({
                "raw": f"[no_transcript] recording_url={recording_url}",
                "parsed": {"step": "reservation", "error": "transcription_failed"},
                "caller": caller,
                "target": "audio",
                "token": CHECK_RESERVE_TOKEN
            })
            logger.info("バックグラウンド: 文字起こし失敗")
            return

        if not user_text:
            return

        parsed = extract_datetime_and_name(user_text, today=datetime.now())
        if not parsed:
            logger.info("バックグラウンド: 日時抽出失敗")
            send_to_gas({"raw": user_text, "parsed": {"step": "reservation", "error": "parse_failed"}, "target": "audio", "token": CHECK_RESERVE_TOKEN})
            return

        date = parsed.get("requested_date")
        timeStart = parsed.get("requested_time")
        name_reading = parsed.get("name_reading")
        name_kanji = parsed.get("name_kanji")

        date, timeStart = validate_and_normalize_datetime(parsed, user_text)
        if not date or not timeStart:
            logger.info("バックグラウンド: 日時不明瞭")
            send_to_gas({"raw": user_text, "parsed": {"step": "reservation", "error": "missing_date_or_time", "parsed": parsed}, "target": "audio", "token": CHECK_RESERVE_TOKEN})
            return

        # 予約チェック（GAS_CHECK_URL）を呼ぶ
        res = try_reserve(date, timeStart, name_reading, name_kanji, caller, user_text)
        if res.get("status") == "error":
            logger.error("バックグラウンド: GAS checkReserve error: %s", res.get("message"))
            send_to_gas({"raw": user_text, "parsed": {"step": "reservation", "error": "gas_error", "detail": res}, "target": "audio", "token": CHECK_RESERVE_TOKEN})
            return

        # 予約確定なら reservations にメタだけ送る（raw は既に audio にある）
        if res.get("reserved"):
            reservation_id = res.get("reservationId")

            # ✅ SMS送信（キャンセルURL付き）
            sms_text = build_sms_text(date, timeStart, reservation_id)
            send_sms(caller, sms_text)

            send_to_gas({
                "raw": None,
                "parsed": {
                    "step": "reservation",
                    "date": date,
                    "time": timeStart,
                    "name_reading": name_reading,
                    "name_kanji": name_kanji,
                    "reservationId": reservation_id
                },
                "target": "reservations",
                "token": CHECK_RESERVE_TOKEN
            })

            logger.info("予約確定＋SMS送信完了")
            confirm_prompt = f"患者に電話で伝える短い確定メッセージを作ってください。日時: {date} {timeStart}、お名前（読み）: {name_reading or '不明' }。出力は一文で。"
            confirm_text = call_ai_via_http(confirm_prompt) or f"では、{date}の{timeStart}でご予約を確定しました。お待ちしています。"
            send_to_gas({
                "raw": None,
                "parsed": {
                    "step": "reservation",
                    "date": date,
                    "time": timeStart,
                    "name_reading": name_reading,
                    "name_kanji": name_kanji,
                    "reservationId": reservation_id
                },
                "target": "reservations",
                "token": CHECK_RESERVE_TOKEN
            })
            logger.info("バックグラウンド: 予約確定処理完了")
        else:
            # 予約不可は reservations に書き込まず audio に理由を残す
            reason = res.get("reason", "unknown")

            # ✅ SMS送信（代替案通知）
            fail_sms = f"申し訳ありません。{date} {timeStart} は既に埋まっています。別の日時でご検討ください。"
            send_sms(caller, fail_sms)

            alt_prompt = f"予約が取れなかったので、別の時間を提案する短い案内を作ってください。元の希望: {date} {timeStart}。"
            alt_text = call_ai_via_http(alt_prompt) or "申し訳ありません、その時間は既に埋まっています。別の時間をお願いします。"

            send_to_gas({
                "raw": None,
                "parsed": {
                    "step": "reservation",
                    "date": date,
                    "time": timeStart,
                    "name_reading": name_reading,
                    "name_kanji": name_kanji,
                    "reserved": False,
                    "reason": reason,
                    "alt_text": alt_text
                },
                "target": "audio",
                "token": CHECK_RESERVE_TOKEN
            })

            logger.info("予約不可＋SMS送信完了")
            logger.info("バックグラウンド: 予約不可処理完了")
    except Exception:
        logger.exception("バックグラウンド予約処理で例外発生")

# -----------------------
# /ai エンドポイント（repeat / ai / reservation / hold）
# -----------------------
@app.route("/ai", methods=["POST"])
def ai():
    logger.info("========== /ai HIT ==========")

    form = request.form or {}
    logger.info("受信データ: %s", form)
    mode = form.get("mode")
    logger.info("mode: %s", mode)

    # Helper: decide whether to return TwiML or JSON
    wants_twiml = False
    accept = request.headers.get("Accept", "")
    user_agent = request.headers.get("User-Agent", "")
    if (
        "text/xml" in accept
        or form.get("twilio_response") == "1"
        or request.args.get("twilio_response") == "1"
        or ("Twilio" in (user_agent or ""))
        or form.get("CallSid")
    ):
        wants_twiml = True

    recording_url = form.get("RecordingUrl")
    caller = form.get("From") or form.get("Caller") or ""

    # ------------------------
    # repeat
    # ------------------------
    if mode == "repeat":
        logger.info("---- repeat mode ----")
        if not recording_url:
            msg = "録音を取得できませんでした。"
            logger.info(msg)
            if wants_twiml:
                return twiml_say(msg)
            return jsonify({"ai_response": msg}), 200

        audio_file = download_recording(recording_url)
        user_text = None
        if audio_file:
            user_text = transcribe_via_http(audio_file)
            try:
                os.remove(audio_file)
            except Exception:
                pass

        if not user_text:
            logger.info("同期で文字起こしできず。バックグラウンドで処理します。")
            threading.Thread(target=background_repeat_process, args=(None, caller), daemon=True).start()
            send_to_gas({"raw": f"[no_transcript] recording_url={recording_url}", "parsed": {"step": "repeat", "error": "transcription_failed"}, "caller": caller, "target": "audio", "token": CHECK_RESERVE_TOKEN})
            msg = "音声を受け取りました。確認後折り返します。"
            if wants_twiml:
                return twiml_say(msg)
            return jsonify({"ai_response": msg, "user_text": None}), 200

        name_kanji, reading = extract_name_and_reading(user_text)
        if reading:
            display_reply = f"{user_text}（読み: {reading}）ですね。"
            tts_text = f"{reading}ですね。"
        else:
            display_reply = f"{user_text}ですね。"
            tts_text = f"{user_text}ですね。"

        send_to_gas({"raw": user_text, "parsed": {"step": "repeat", "reading": reading}, "caller": caller, "target": "audio", "token": CHECK_RESERVE_TOKEN})

        threading.Thread(target=background_repeat_process, args=(user_text, caller), daemon=True).start()

        if wants_twiml:
            return twiml_say(tts_text)
        return jsonify({"ai_response": display_reply, "tts_text": tts_text, "user_text": user_text}), 200

    # ------------------------
    # ai (相談応答)
    # ------------------------
    if mode == "ai":
        if not recording_url:
            msg = "録音データがありません。"
            if wants_twiml:
                return twiml_say(msg)
            return jsonify({"ai_response": msg}), 200

        audio_file = download_recording(recording_url)
        user_text = None
        if audio_file:
            user_text = transcribe_via_http(audio_file)
            try:
                os.remove(audio_file)
            except Exception:
                pass

        if not user_text:
            logger.info("同期で文字起こしできず。バックグラウンドで処理します。")
            threading.Thread(target=lambda: send_to_gas({"raw": None, "parsed": {"step": "ai", "error": "transcription_failed"}, "target": "audio", "token": CHECK_RESERVE_TOKEN}), daemon=True).start()
            msg = "音声を受け取りました。確認後折り返します。"
            if wants_twiml:
                return twiml_say(msg)
            return jsonify({"ai_response": msg, "user_text": None}), 200

        prompt = f"""あなたは実在するクリニックの受付スタッフです。
電話応対として自然でやわらかい日本語で話してください。
マニュアル的な表現は禁止です。
患者の相談に対して、安心感のある短い案内をしてください。
相談内容:
{user_text}
"""
        reply = call_ai_via_http(prompt) or "承知しました。"
        name_kanji, reading = extract_name_and_reading(user_text)
        parsed = {"step": "ai", "reply": reply}
        if name_kanji:
            parsed["name_kanji"] = name_kanji
        if reading:
            parsed["reading"] = reading

        threading.Thread(target=send_to_gas, args=({"raw": user_text, "parsed": parsed, "target": "audio", "token": CHECK_RESERVE_TOKEN},), daemon=True).start()

        if wants_twiml:
            return twiml_say(reply)
        return jsonify({"ai_response": reply, "user_text": user_text}), 200

    # ------------------------
    # hold / slot（候補だけ抑える）
    # ------------------------
    if mode == "hold" or mode == "slot":
        if not recording_url:
            msg = "録音を取得できませんでした。"
            if wants_twiml:
                return twiml_say(msg)
            return jsonify({"ai_response": msg}), 200

        audio_file = download_recording(recording_url)
        user_text = None
        if audio_file:
            user_text = transcribe_via_http(audio_file)
            try:
                os.remove(audio_file)
            except Exception:
                pass

        if not user_text:
            user_text = f"[no_transcript] recording_url={recording_url}"

        parsed = None
        if user_text:
            parsed = extract_datetime_and_name(user_text, today=datetime.now())

        send_to_gas({"raw": user_text, "parsed": {"step": "hold", "parsed": parsed}, "caller": caller, "target": "audio", "token": CHECK_RESERVE_TOKEN})

        send_to_gas({
            "raw": user_text,
            "parsed": {
                "step": "slots",
                "date": parsed.get("requested_date") if parsed else "",
                "timeStart": parsed.get("requested_time") if parsed else "",
                "name_reading": parsed.get("name_reading") if parsed else ""
            },
            "caller": caller,
            "target": "slots",
            "token": CHECK_RESERVE_TOKEN
        })

        msg = "候補を確保しました。確認後ご連絡します。"
        if wants_twiml:
            return twiml_say(msg)
        return jsonify({"ai_response": msg, "user_text": user_text}), 200

    # ------------------------
    # reservation (自動予約フロー)
    # ------------------------
    if mode == "reservation":
        if not recording_url:
            msg = "録音を取得できませんでした。"
            if wants_twiml:
                return twiml_say(msg)
            return jsonify({"ai_response": msg}), 200

        # If Twilio expects TwiML, respond immediately and process in background to avoid timeouts
        if wants_twiml:
            threading.Thread(target=process_reservation_background, args=(recording_url, caller), daemon=True).start()
            return twiml_say("音声を受け取りました。確認後、折り返しご連絡します。")

        # Non-Twilio or API clients: perform synchronous processing (same as background function)
        audio_file = download_recording(recording_url)
        user_text = None
        if audio_file:
            user_text = transcribe_via_http(audio_file)
            try:
                os.remove(audio_file)
            except Exception:
                pass

        if not user_text:
            logger.info("同期で文字起こしできず。バックグラウンドで処理します。")
            threading.Thread(target=lambda: send_to_gas({"raw": f"[no_transcript] recording_url={recording_url}", "parsed": {"step": "reservation", "error": "transcription_failed"}, "target": "audio", "token": CHECK_RESERVE_TOKEN}), daemon=True).start()
            msg = "音声を受け取りました。確認後折り返します。"
            return jsonify({"ai_response": msg, "user_text": None}), 200

        # まず audio に生ログを送る
        parsed = extract_datetime_and_name(user_text, today=datetime.now())
        send_to_gas({"raw": user_text, "parsed": {"step": "reservation", "parsed": parsed}, "caller": caller, "target": "audio", "token": CHECK_RESERVE_TOKEN})

        if not parsed:
            threading.Thread(target=send_to_gas, args=({"raw": user_text, "parsed": {"step": "reservation", "error": "parse_failed"}, "target": "audio", "token": CHECK_RESERVE_TOKEN},), daemon=True).start()
            msg = "すみません、日時がはっきり聞き取れませんでした。折り返します。"
            return jsonify({"ai_response": msg, "user_text": user_text}), 200

        # ここから時刻再抽出と保留ロジック
        date = parsed.get("requested_date")
        timeStart = parsed.get("requested_time")
        name_reading = parsed.get("name_reading")
        name_kanji = parsed.get("name_kanji")

        # 正規化（簡易）
        date, timeStart = validate_and_normalize_datetime(parsed, user_text)
        if not date or not timeStart:
            logger.info("同期処理: 日時不明瞭")
            send_to_gas({"raw": user_text, "parsed": {"step": "reservation", "error": "missing_date_or_time", "parsed": parsed}, "target": "audio", "token": CHECK_RESERVE_TOKEN})
            return jsonify({"ai_response": "日時が不明瞭でした。折り返します。", "user_text": user_text}), 200

        # 予約チェック（GAS_CHECK_URL）を呼ぶ
        res = try_reserve(date, timeStart, name_reading, name_kanji, caller, user_text)
        if res.get("status") == "error":
            logger.error("同期処理: GAS checkReserve error: %s", res.get("message"))
            send_to_gas({"raw": user_text, "parsed": {"step": "reservation", "error": "gas_error", "detail": res}, "target": "audio", "token": CHECK_RESERVE_TOKEN})
            return jsonify({"status": "error", "message": "reservation_check_failed"}), 500

        if res.get("reserved"):
            reservation_id = res.get("reservationId")

            # ✅ SMS送信（キャンセルURL付き）
            sms_text = build_sms_text(date, timeStart, reservation_id)
            send_sms(caller, sms_text)

            # reservations にメタを送信（raw は audio に既にある）
            send_to_gas({
                "raw": None,
                "parsed": {
                    "step": "reservation",
                    "date": date,
                    "time": timeStart,
                    "name_reading": name_reading,
                    "name_kanji": name_kanji,
                    "reservationId": reservation_id
                },
                "target": "reservations",
                "token": CHECK_RESERVE_TOKEN
            })

            msg = f"ご予約を承りました。日時: {date} {timeStart}、お名前: {name_reading or name_kanji or '不明'}"
            return jsonify({"status": "ok", "reserved": True, "message": msg, "reservationId": reservation_id}), 200
        else:
            reason = res.get("reason", "unknown")
            alt_prompt = f"予約が取れなかったので、別の時間を提案する短い案内を作ってください。元の希望: {date} {timeStart}。"
            alt_text = call_ai_via_http(alt_prompt) or "申し訳ありません、その時間は既に埋まっています。別の時間をお願いします。"

            # 予約不可は audio に理由を残す
            send_to_gas({
                "raw": None,
                "parsed": {
                    "step": "reservation",
                    "date": date,
                    "time": timeStart,
                    "name_reading": name_reading,
                    "name_kanji": name_kanji,
                    "reserved": False,
                    "reason": reason,
                    "alt_text": alt_text
                },
                "target": "audio",
                "token": CHECK_RESERVE_TOKEN
            })

            # 代替案 SMS を送る（任意）
            fail_sms = f"申し訳ありません。{date} {timeStart} は既に埋まっています。別の日時でご検討ください。"
            send_sms(caller, fail_sms)

            return jsonify({"status": "ok", "reserved": False, "message": alt_text}), 200

    # default
    return jsonify({"status": "ok", "message": "no_action"}), 200

# -----------------------
# ルート（動作確認用）
# -----------------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "reservation server running"}), 200

# -----------------------
# サーバ起動
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse

app = Flask(__name__)

@app.route("/voice", methods=['POST'])
def voice():
    print("電話リクエスト届いた:", request.form)
    resp = VoiceResponse()
    resp.say("テスト通話成功。Renderからの電話です。", voice="alice", language="ja-JP")
    return Response(str(resp), mimetype="text/xml")

@app.route("/")
def home():
    return "Server is running"

if __name__ == "__main__":
    app.run()
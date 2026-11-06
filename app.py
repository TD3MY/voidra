import base64
import logging
import os
import re

import requests
from flask import Flask, jsonify, request, send_from_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("voidra-api")

# ---- Config (set these as Environment Variables on Render) -----------------
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "https://td3my.github.io")
CF_ACCOUNTS = [
    {"id": os.getenv("CLOUDFLARE_ACCOUNT_ID"), "token": os.getenv("CLOUDFLARE_API_TOKEN")},
    {"id": os.getenv("CLOUDFLARE_ACCOUNT_ID2"), "token": os.getenv("CLOUDFLARE_API_TOKEN2")},
]
CF_ACCOUNTS = [a for a in CF_ACCOUNTS if a["id"] and a["token"]]

STT_MODEL = "@cf/openai/whisper-large-v3-turbo"
# Fast models first: a voice assistant must answer quickly.
CHAT_MODELS = [
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/meta/llama-4-scout-17b-16e-instruct",
]
STT_TIMEOUT = 40
CHAT_TIMEOUT = 25
MAX_HISTORY = 8

SYSTEM_PROMPT = (
    "You are Voidra, a friendly voice assistant. Reply in the same language the user speaks. "
    "Keep answers short (1-3 sentences) because they are read aloud. "
    "Do not use Markdown, lists, or emojis."
)

history = []  # simple shared history (personal use)

app = Flask(__name__)


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


def cf_run(model, payload, timeout):
    """Call a Cloudflare Workers AI model, trying each account. Returns result dict or None."""
    for acc in CF_ACCOUNTS:
        try:
            r = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{acc['id']}/ai/run/{model}",
                headers={"Authorization": f"Bearer {acc['token']}"},
                json=payload,
                timeout=timeout,
            )
            data = r.json()
            if data.get("success"):
                return data.get("result") or {}
            logger.warning("%s failed: %s", model, data.get("errors"))
        except Exception as e:
            logger.error("%s exception: %s", model, e)
    return None


def transcribe(audio_bytes):
    result = cf_run(STT_MODEL, {"audio": base64.b64encode(audio_bytes).decode()}, STT_TIMEOUT)
    return ((result or {}).get("text") or "").strip()


def clean(text):
    text = re.sub(r"[`*_#>]+", "", text or "")
    return text.strip()


def ask(user_text):
    history.append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-MAX_HISTORY:]
    for model in CHAT_MODELS:
        result = cf_run(model, {"messages": messages}, CHAT_TIMEOUT)
        if not result:
            continue
        reply = result.get("response")
        if not reply:
            choices = result.get("choices") or [{}]
            reply = (choices[0].get("message") or {}).get("content", "")
        reply = clean(reply)
        if reply:
            history.append({"role": "assistant", "content": reply})
            del history[:-MAX_HISTORY]
            return reply
    history.pop()  # drop the unanswered question
    return None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health")
def health():
    return "ok"


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def api_chat():
    if request.method == "OPTIONS":
        return "", 204
    f = request.files.get("audio")
    if not f:
        return jsonify(error="no audio received"), 400
    if not CF_ACCOUNTS:
        return jsonify(error="server has no Cloudflare credentials"), 500
    try:
        user_text = transcribe(f.read())
        if not user_text:
            return jsonify(error="could not transcribe audio"), 422
        logger.info("STT ok: %s", user_text[:60])
        reply = ask(user_text)
        if not reply:
            return jsonify(error="AI models not responding"), 502
        logger.info("AI reply ready")
        return jsonify(user_text=user_text, bot_reply=reply)
    except Exception as e:
        logger.error("api_chat error: %s", e)
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))

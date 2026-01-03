import re
import requests
from flask import Flask, render_template, request, session

# ==========================
#   APP CONFIG
# ==========================
BOT_NAME = "Rewordly"

# ==========================
#   OLLAMA CONFIG (LOCAL)
# ==========================
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:4b"  # change to "mistral" if you want faster

# ==========================
#        FLASK APP
# ==========================
app = Flask(__name__)
app.secret_key = "change-this-to-a-random-secret"


# --------------------------
# Helpers
# --------------------------
def format_bot_text(text: str) -> str:
    """Convert **bold** and newlines into HTML for the chat UI."""
    if not text:
        return ""
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return text.replace("\n", "<br>")


def strip_html(s: str) -> str:
    """Remove HTML tags for clean context sent to model."""
    if not s:
        return ""
    return re.sub(r"<.*?>", "", s)


def build_context(history, limit: int = 5) -> str:
    """Short context = faster responses."""
    recent = history[-limit:]
    lines = []
    for role, msg in recent:
        speaker = "You" if role == "user" else BOT_NAME
        lines.append(f"{speaker}: {strip_html(msg)}")
    return "\n".join(lines)


def is_rewrite_intent(text: str) -> bool:
    t = text.lower().strip()

    command_prefixes = ["rephrase:", "rewrite:", "professional:", "casual:", "concise:", "fix:", "polish:"]
    if any(t.startswith(p) for p in command_prefixes):
        return True

    keywords = [
        "rephrase", "rewrite", "paraphrase", "reword", "word this",
        "make it professional", "make it formal", "make it casual",
        "sound better", "improve this", "polish", "fix grammar",
        "correct this", "clean this up", "clarify", "clearer", "shorten",
        "summarize", "make it concise", "tone", "more friendly", "more confident"
    ]
    return any(k in t for k in keywords)


def build_rewordly_prompt(history, user_msg: str) -> str:
    context = build_context(history, limit=5)
    rewrite_requested = is_rewrite_intent(user_msg)

    return f"""
You are {BOT_NAME}, a friendly conversational AI writing assistant.

Rules:
- If the user wants rewriting/rephrasing, return three versions: Professional, Casual, Concise.
- If they ask to rewrite but did NOT paste text, ask them to paste it.
- Otherwise respond normally and helpfully.

When rewriting, output EXACTLY:
**Professional:**
...

**Casual:**
...

**Concise:**
...

Context:
{context}

User:
{user_msg}

Rewrite intent detected: {rewrite_requested}
""".strip()


def call_ollama(prompt: str) -> str:
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You are {BOT_NAME}, a friendly writing assistant. "
                        "Help rephrase text, improve clarity, adjust tone, and fix grammar. "
                        "Be natural and not robotic."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            # speed + shorter answers (optional but recommended)
            "options": {
                "temperature": 0.3,
                "num_predict": 220
            }
        }

        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        text = (data.get("message", {}).get("content") or "").strip()
        return text if text else "I didn’t get a response. Please try again."

    except Exception as e:
        print("\n[Ollama ERROR FULL]")
        print(str(e))
        print("[END ERROR]\n")
        return (
            f"⚠️ {BOT_NAME} can’t reach Ollama right now.<br>"
            "Make sure the Ollama app is running and the model is installed, then try again."
        )


# ==========================
#          ROUTE
# ==========================
@app.route("/", methods=["GET", "POST"])
def chat():
    if "history" not in session:
        session["history"] = []

    history = session["history"]

    if request.method == "GET" and not history:
        greeting = (
            f"**{BOT_NAME}:** Hi! 👋 What’s your goal today—rephrasing, making something more professional, "
            "or improving clarity?\n"
            "Tip: paste a sentence/paragraph and tell me the tone you want."
        )
        history.append(("bot", format_bot_text(greeting)))
        session["history"] = history

    if request.method == "POST":
        action = request.form.get("action", "send")

        if action == "reset":
            history = []
            greeting = f"**{BOT_NAME}:** Starting a new conversation ✅\nWhat would you like help with today?"
            history.append(("bot", format_bot_text(greeting)))
            session["history"] = history
            return render_template("chat.html", history=history, bot_name=BOT_NAME)

        user_msg = request.form.get("message", "").strip()
        if user_msg:
            history.append(("user", format_bot_text(user_msg)))

            prompt = build_rewordly_prompt(history, user_msg)
            raw_reply = call_ollama(prompt)

            history.append(("bot", format_bot_text(raw_reply)))
            session["history"] = history

    return render_template("chat.html", history=history, bot_name=BOT_NAME)


if __name__ == "__main__":
    print(f"\n{BOT_NAME} running (Ollama Local)!")
    print("Chat UI → http://127.0.0.1:5000/\n")
    app.run(debug=True, use_reloader=False)

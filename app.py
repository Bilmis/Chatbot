from flask import Flask, request, jsonify
import requests
import os
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Hugging Face model URL for Mixtral
MODEL_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
HF_API_KEY = os.getenv("HF_API_KEY")
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# PostgreSQL connection
PG_CONN_STRING = os.getenv("POSTGRES_URL")

def get_db_connection():
    return psycopg2.connect(PG_CONN_STRING, cursor_factory=RealDictCursor)

def save_message(session_id, role, message):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_memory (session_id, role, message)
                VALUES (%s, %s, %s);
            """, (session_id, role, message))
        conn.commit()

def get_chat_history(session_id, limit=6):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT role, message FROM chat_memory
                WHERE session_id = %s
                ORDER BY timestamp DESC
                LIMIT %s;
            """, (session_id, limit))
            return list(reversed(cur.fetchall()))

# Format prompt for Mixtral-style chat
def build_prompt(history):
    prompt = ""
    for entry in history:
        prompt += f"{entry['role'].capitalize()}: {entry['message']}\n"
    prompt += "Assistant:"
    return prompt

@app.route("/")
def home():
    return jsonify({"status": "Chatbot backend is live."})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "").strip()
        session_id = data.get("session_id", "").strip()

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400

        # Save user message
        save_message(session_id, "user", prompt)

        # Build prompt context from memory
        history = get_chat_history(session_id)
        context = build_prompt(history)

        payload = {
            "inputs": context,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "return_full_text": False
            },
            "options": {
                "use_cache": False,
                "wait_for_model": True
            }
        }

        response = requests.post(MODEL_URL, headers=headers, json=payload)

        if response.status_code == 200:
            generated = response.json()
            assistant_reply = generated[0]["generated_text"].strip()

            save_message(session_id, "assistant", assistant_reply)

            return jsonify({
                "response": assistant_reply
            })
        else:
            return jsonify({"error": response.text}), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

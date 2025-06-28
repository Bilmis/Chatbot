from flask import Flask, request, jsonify
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Gemini setup (Client-style)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

# Mixtral (Hugging Face fallback)
HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
hf_headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# PostgreSQL (Neon)
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

def build_prompt(history):
    prompt = ""
    for entry in history:
        prompt += f"{entry['role'].capitalize()}: {entry['message']}\n"
    prompt += "Assistant:"
    return prompt

@app.route("/")
def home():
    return jsonify({"status": "Chatbot backend is live with Gemini Client + Mixtral fallback."})

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

        # Save user message to memory
        save_message(session_id, "user", prompt)

        # Get chat history and build context
        history = get_chat_history(session_id)
        context = build_prompt(history)

        # Try Gemini 1.5 Flash via client
        try:
            gemini_response = gemini_model.generate_content(context)
            reply = gemini_response.text.strip()
            save_message(session_id, "assistant", reply)
            return jsonify({"response": reply})
        except Exception as gemini_error:
            print("Gemini failed:", str(gemini_error))

        # Fallback to Mixtral
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

        response = requests.post(MODEL_URL, headers=hf_headers, json=payload)
        if response.status_code == 200:
            mixtral_output = response.json()
            fallback_reply = mixtral_output[0]["generated_text"].strip()
            save_message(session_id, "assistant", fallback_reply)
            return jsonify({"response": fallback_reply})
        else:
            return jsonify({"error": "Both Gemini and Mixtral failed."}), 500

    except Exception as ex:
        return jsonify({"error": str(ex)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

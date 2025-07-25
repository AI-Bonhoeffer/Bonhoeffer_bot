from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
#from twilio.rest import Client
import os
import re
import time
from db import load_vector_store

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super-secret")  # For session handling

verified_users = {}  # Stores user_id -> expiry timestamp
vector_store = load_vector_store()

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
    retriever=vector_store.as_retriever()
)

# 🔄 Refresh route to clear messages
@app.route("/refresh-chat", methods=["GET"])
def refresh_chat():
    session.pop("messages", None)
    return redirect(url_for("index"))

# Shared input processing function
def process_user_input(user_input, user_id):
    responses = []
    current_time = time.time()

    # Check verification status
    is_verified = user_id in verified_users and current_time < verified_users[user_id]

    if "7320811109" in user_input and "123456" in user_input:
        verified_users[user_id] = current_time + 86400  # 24 hours
        responses.append("✅ You are verified. Valid for 24 hours.")

    elif "7320811109" in user_input or "123456" in user_input:
        responses.append("❌ Wrong ID or password. Please enter correct credentials or contact the sales team.")

    elif len(user_input.strip()) == 4 and user_input.strip().isalnum():
        if not is_verified:
            responses.append("🔒 Please enter your ID and password to access price information.")
        else:
            query = f"What is the price of model ending with {user_input}?"
            reply = qa_chain.run(query)
            responses.append(reply)

    elif any(word in user_input.lower() for word in ["production time", "lead time", "manufacturing time"]):
        responses.append("🏭 The production time for any model is **90 days**.")

    elif any(word in user_input.lower() for word in ["price", "cost", "rate", "paisa", "pice", "rupees", "rupee"]):
        if not is_verified:
            responses.append("🔒 Please enter your ID and password to access price information.")
        else:
            match = re.search(r"\b([A-Za-z0-9]{4})\b", user_input)
            if match:
                code = match.group(1)
                query = f"What is the price of model ending with {code}?"
                reply = qa_chain.run(query)
            else:
                reply = qa_chain.run(user_input)
            responses.append(reply)

    elif any(word in user_input.lower() for word in ["invoice", "packaging list", "dispatch", "packing"]):
        if not is_verified:
            responses.append("🔒 Please enter your ID and password to access packaging or invoice information.")
        else:
            reply = qa_chain.run(user_input)
            responses.append(reply)

    else:
        reply = qa_chain.run(user_input)
        responses.append(reply)

    return responses, is_verified

# 🌐 Web UI chat route
@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = [{
            # "role": "assistant",
            # "content": "👋 Welcome to Bonhoeffer Bot! How can I assist you today?"
        }]

    if request.method == "POST":
        user_input = request.form["message"]
        session["messages"].append({"role": "user", "content": user_input})
        user_id = request.remote_addr  # Use IP address for web sessions

        replies, _ = process_user_input(user_input, user_id)
        for reply in replies:
            session["messages"].append({"role": "assistant", "content": reply})

    return render_template("index.html", messages=session["messages"])


@app.route("/webhook", methods=["GET", "POST"])
def meta_webhook():
    if request.method == "GET":
        VERIFY_TOKEN = "test"  # Any string you choose
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")

        if mode == "subscribe" and token == VERIFY_TOKEN:
            return challenge, 200
        return "Verification failed", 403

    elif request.method == "POST":
        data = request.get_json()
        print("📨 Incoming message:", data)

        try:
            entry = data["entry"][0]
            changes = entry["changes"][0]
            value = changes["value"]

            messages = value.get("messages")
            if messages:
                msg = messages[0]
                user_id = msg["from"]
                user_input = msg["text"]["body"]

                replies, _ = process_user_input(user_input, user_id)

                for reply in replies:
                    send_whatsapp_message_meta(user_id, reply)

        except Exception as e:
            print("❌ Error:", e)

        return "OK", 200


def send_whatsapp_message_meta(to_number, message):
    import requests, os

    token = os.getenv("META_WA_ACCESS_TOKEN")
    phone_number_id = os.getenv("META_PHONE_NUMBER_ID")

    url = f"https://graph.facebook.com/v18.0/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": { "body": message }
    }

    response = requests.post(url, headers=headers, json=data)
    print("📤 Sent:", response.status_code, response.text)


if __name__ == "__main__":
    
    app.run(debug=True, port=5050)

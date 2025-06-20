from flask import Flask, render_template, request, session, redirect, url_for
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from twilio.twiml.messaging_response import MessagingResponse
import os
import re
import time
from db import load_vector_store

# No need for load_dotenv since we're hardcoding
openai_api_key = "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"
print("✅ OpenAI API Key Loaded")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super-secret")

verified_users = {}
vector_store = load_vector_store()

qa_chain = RetrievalQA.from_chain_type(
   llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_api_key),

    retriever=vector_store.as_retriever()
)

@app.route("/refresh-chat", methods=["GET"])
def refresh_chat():
    session.pop("messages", None)
    return redirect(url_for("index"))

def process_user_input(user_input, user_id):
    responses = []
    current_time = time.time()
    is_verified = user_id in verified_users and current_time < verified_users[user_id]

    if "7320811109" in user_input and "123456" in user_input:
        verified_users[user_id] = current_time + 86400
        responses.append("✅ You are verified. Valid for 24 hours.")
    elif "7320811109" in user_input or "123456" in user_input:
        responses.append("❌ Wrong ID or password.")
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
            responses.append("🔒 Please enter your ID and password to access invoice/packing details.")
        else:
            reply = qa_chain.run(user_input)
            responses.append(reply)
    else:
        reply = qa_chain.run(user_input)
        responses.append(reply)

    return responses, is_verified

@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = []

    if request.method == "POST":
        user_input = request.form["message"]
        session["messages"].append({"role": "user", "content": user_input})
        user_id = request.remote_addr

        replies, _ = process_user_input(user_input, user_id)
        for reply in replies:
            session["messages"].append({"role": "assistant", "content": reply})

    return render_template("chat.html", messages=session["messages"])

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.values.get('Body', '').strip()
    user_id = request.values.get('From', 'unknown')

    resp = MessagingResponse()
    if not incoming_msg:
        resp.message("⚠️ Sorry, I didn't get your message.")
        return str(resp)

    replies, _ = process_user_input(incoming_msg, user_id)
    for reply in replies:
        resp.message(reply)

    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)

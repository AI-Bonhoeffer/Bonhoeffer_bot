from flask import Flask, render_template, request, session, redirect, url_for
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import re

from db import load_vector_store

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
 
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session

# Load the vector store
vector_store = load_vector_store()

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
    retriever=vector_store.as_retriever()
)

@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = []
    if "user_verified" not in session:
        session["user_verified"] = False

    messages = session["messages"]
    user_verified = session["user_verified"]

    if request.method == "POST":
        user_input = request.form["message"]
        messages.append({"role": "user", "content": user_input})

        if "7320811109" in user_input and "123456" in user_input:
            user_verified = True
            messages.append({
                "role": "assistant",
                "content": "✅ You are verified. Valid for 24 hours."
            })

        elif any(x in user_input for x in ["7320811109", "123456"]) and not user_verified:
            messages.append({
                "role": "assistant",
                "content": "❌ Please enter correct ID and password."
            })

        elif any(keyword in user_input.lower() for keyword in ["price", "cost", "rate", "paisa", "money", "dollar", "rs", "rupees", "invoice", "invoci", "invic", "invouc", "invoic", "moq", "maq", "meq","mooq","muq"]):
            if not user_verified:
                messages.append({
                    "role": "assistant",
                    "content": "🔒 Please enter your ID and password to access price information."
                })
            else:
                match = re.search(r"\b([A-Za-z0-9]{4})\b", user_input)
                if match:
                    code = match.group(1)
                    query = f"What is the price of model ending with {code}?"
                    bot_reply = qa_chain.run(query)
                else:
                    bot_reply = qa_chain.run(user_input)

                messages.append({"role": "assistant", "content": bot_reply})

        elif any(word in user_input.lower() for word in ["invoice", "packaging list", "dispatch", "packing"]):
            if not user_verified:
                messages.append({
                    "role": "assistant",
                    "content": "🔒 Please enter your ID and password to access packaging or invoice information."
                })
            else:
                bot_reply = qa_chain.run(user_input)
                messages.append({"role": "assistant", "content": bot_reply})

        elif len(user_input.strip()) == 4 and user_input.strip().isalnum():
            if not user_verified:
                messages.append({
                    "role": "assistant",
                    "content": "🔒 Please enter your ID and password to access price information."
                })
            else:
                query = f"What is the price of model ending with {user_input}?"
                bot_reply = qa_chain.run(query)
                messages.append({"role": "assistant", "content": bot_reply})

        elif any(word in user_input.lower() for word in ["production time", "lead time", "manufacturing time"]):
            messages.append({
                "role": "assistant",
                "content": "🏭 The production time for any model is **90 days**."
            })

        else:
            bot_reply = qa_chain.run(user_input)
            messages.append({"role": "assistant", "content": bot_reply})

        session["messages"] = messages
        session["user_verified"] = user_verified

    return render_template("chat.html", messages=messages)

@app.route("/refresh")
def refresh():
    session.pop("messages", None)
    session.pop("user_verified", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
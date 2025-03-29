from flask import Flask, request, jsonify, render_template
from src.components.chatbot import ChatBot
import os
from dotenv import load_dotenv

load_dotenv()
# Explicitly load the API key
api_key = os.getenv("API_KEY")
print(api_key)
if api_key is None:
    raise ValueError("API_KEY is not set. Please export it before running the app.")

app = Flask(__name__)

# Serve the HTML page
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    
    user_input = data.get("instruction", "").strip()
    if not user_input:
        return jsonify({"response": "Please enter a valid question.",
                        "category": "Unknown",
                        "intent": "Unknown"})

    chatbot = ChatBot(instruction=user_input)

    try:
        category = chatbot.pred_category() or "Unknown"
    except Exception as e:
        category = "Unknown"
        print(f"Category Prediction Error: {str(e)}")  # Log for debugging

    try:
        intent = chatbot.pred_intent() or "Unknown"
    except Exception as e:
        intent = "Unknown"
        print(f"Intent Prediction Error: {str(e)}")  # Log for debugging

    try:
        faiss_response = chatbot.retrieve_faiss_response()
        retrieved_response = faiss_response["response"] if faiss_response else "No relevant match found."
    except Exception as e:
        retrieved_response = "Error retrieving FAISS response."
        print(f"FAISS Retrieval Error: {str(e)}")  # Log for debugging

    try:
        response = chatbot.generate_response() or "I'm sorry, I couldn't process that request."
    except Exception as e:
        response = "I'm sorry, something went wrong."
        print(f"Response Generation Error: {str(e)}")  # Log for debugging

    return jsonify({
        "response": response,
        "category": category,
        "intent": intent,
        "faiss_response": retrieved_response
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  
    app.run(host="0.0.0.0", port=port, debug=True)

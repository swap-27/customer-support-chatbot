from utils import load_object
import os
import requests
import time
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from dotenv import load_dotenv


class ChatBot:
    def __init__(self, instruction: str):
        self.instruction = instruction
        #self.api_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

        load_dotenv()
        self.api_key = os.getenv("API_KEY")  # Fetch API Key from env
        self.headers = {"Authorization": f"Bearer {self.api_key}"}  # API Key for authentication
        
        self.preprocessor_category_obj_file_path = os.path.join('artifacts', 'vectorizer_category.pkl')
        self.preprocessor_intent_obj_file_path = os.path.join('artifacts', 'vectorizer_intent.pkl')
        self.category_model_path = os.path.join('artifacts', 'RandomForest_Category.pkl')
        self.intent_model_path = os.path.join('artifacts', 'RandomForest_Intent.pkl')

        # Lazy loading
        self.category_model = None
        self.intent_model = None
        self.vectorizer_category = None
        self.vectorizer_intent = None

        # Predictions stored once calculated
        self.category_pred = None
        self.intent_pred = None

        # FAISS Setup
        self.faiss_index_path = "faiss/faiss_index.bin"
        self.responses_path = "faiss/responses.pkl"

        self.faiss_index = None
        self.faiss_text = None
        self.responses = None
        self.model = None

    def pred_category(self):
        if self.category_model is None:
            self.category_model = load_object(self.category_model_path)
            self.vectorizer_category = load_object(self.preprocessor_category_obj_file_path)

        val = self.vectorizer_category.transform([self.instruction])
        self.category_pred = self.category_model.predict(val)[0]
        return self.category_pred

    def pred_intent(self):
        if self.intent_model is None:
            self.intent_model = load_object(self.intent_model_path)
            self.vectorizer_intent = load_object(self.preprocessor_intent_obj_file_path)

        val = self.vectorizer_intent.transform([self.instruction])
        self.intent_pred = self.intent_model.predict(val)[0]
        return self.intent_pred

    def load_faiss(self):
        """Lazy-load FAISS components."""
        if self.faiss_index is None:
            self.faiss_index = faiss.read_index(self.faiss_index_path)

        if self.responses is None:
            with open(self.responses_path, "rb") as f:
                self.responses = pickle.load(f)

        if self.model is None:
            cache_path = "artifacts/model_cache"
            if os.path.exists(cache_path):
                self.model = SentenceTransformer("artifacts/model_cache/all-MiniLM-L6-v2")
            else:
                self.model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_path)

    def retrieve_faiss_response(self):
        """Retrieve the most relevant response from FAISS."""
        self.load_faiss()

        query_embedding = self.model.encode([self.instruction], normalize_embeddings=True)
        distances, indices = self.faiss_index.search(query_embedding, 1)  # Retrieve top-1 match

        idx = indices[0][0]
        return self.responses[idx] if idx >= 0 else None

    def construct_mistral_prompt(self):
        if self.category_pred is None:
            self.pred_category()
        if self.intent_pred is None:
            self.pred_intent()

        faiss_result = self.retrieve_faiss_response()
        self.faiss_text = faiss_result["response"] if faiss_result else "No relevant match found."

        prompt = f"""
        You are an AI-powered customer support assistant. Your goal is to provide professional, helpful, and empathetic responses to user queries.

        ### User's Query:
        "{self.instruction}"

        ### Context:
        - **Predicted Category:** {self.category_pred}
        - **Predicted Intent:** {self.intent_pred}
        - **Similar Query Response (from FAISS):** "{self.faiss_text}"

        ### Instructions:
        - Analyze the user’s query along with the predicted category and intent.
        - Use the FAISS response as a reference, improving it if necessary.
        - Ensure the response is **clear, concise, professional, and helpful**.
        - Maintain an empathetic and customer-friendly tone.
        - Address the user’s concern directly and offer additional guidance if needed.

        ### Expected Output:
        Provide a well-structured response that directly addresses the user’s question while maintaining a professional and helpful tone.
        ### Response:
        """

        return prompt
    
    def generate_response(self):
        prompt = self.construct_mistral_prompt()
        data = {"inputs": prompt}

        for attempt in range(3):  # Retry up to 3 times if rate-limited
            response = requests.post(self.api_url, headers=self.headers, json=data)

            print(f"Response Status Code: {response.status_code}")
            print(f"Response JSON: {response.json()}")

            if response.status_code == 200:
                if response.json()[0]["generated_text"] == "I'm sorry, I can't help you with that. I'm just looking":
                    return self.faiss_text
                else:    
                    generated_text = response.json()[0]["generated_text"]

                    match = re.search(r"Response:\s*(.*)", generated_text, re.DOTALL)
                    if match:
                        actual_response = match.group(1).strip()
                    else:
                        actual_response = generated_text.split("Response:\n         ")[1]
                    

                    return actual_response

            elif response.status_code == 429:  # Too Many Requests
                wait_time = int(response.headers.get("Retry-After", 10))  # Default wait 10 sec
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

            else:
                return f"Error: {response.status_code}, {response.json()}"

        return "Sorry, the chatbot is temporarily unavailable due to API rate limits. Please try again later."

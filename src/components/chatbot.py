from utils import load_object
import os
import requests
import time

class ChatBot:
    def __init__(self, instruction: str):
        self.instruction = instruction
        self.api_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"

        self.api_key = os.getenv("HF_API_KEY")  # Fetch API Key from env
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

    def construct_mistral_prompt(self):
        if self.category_pred is None:
            self.pred_category()
        if self.intent_pred is None:
            self.pred_intent()

        prompt = f"""
        You are a professional AI customer support assistant. Your job is to assist users with their queries.

        User's Query: "{self.instruction}"
        Predicted Category: {self.category_pred}
        Predicted Intent: {self.intent_pred}

        Based on this information, provide a professional and helpful response.
        """
        return prompt
    
    def generate_response(self):
        prompt = self.construct_mistral_prompt()
        data = {"inputs": prompt}

        print(f"\n Sending API Request to: {self.api_url}")
        print(f" Headers: {self.headers}")
        print(f"Payload: {data}")

        for attempt in range(3):  # Retry up to 3 times if rate-limited
            response = requests.post(self.api_url, headers=self.headers, json=data)

            print(f"🔹 Response Status Code: {response.status_code}")
            print(f"🔹 Response JSON: {response.json()}")

            if response.status_code == 200:
                return response.json()[0]["generated_text"]

            elif response.status_code == 429:  # Too Many Requests
                wait_time = int(response.headers.get("Retry-After", 10))  # Default wait 10 sec
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

            else:
                return f"Error: {response.status_code}, {response.json()}"

        return "Sorry, the chatbot is temporarily unavailable due to API rate limits. Please try again later."

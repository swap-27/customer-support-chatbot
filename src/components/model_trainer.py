from sklearn.ensemble import RandomForestClassifier
import os
import sys
from sklearn.metrics import accuracy_score
from utils import save_object
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.logger import logging

class ModelTrainer:
    def __init__(self):
        self.model_category_save_path = os.path.join('artifacts','RandomForest_Category.pkl')
        self.model_intent_save_path = os.path.join('artifacts','RandomForest_Intent.pkl')

    def model_trainer_category(self, X_train_category, X_test_category, y_train_category, y_test_category):
        try:    
            logging.info("Initializing Category model loading, fitting and saving.")
            category_model = RandomForestClassifier(class_weight='balanced', random_state=42)
            category_model.fit(X_train_category, y_train_category)
            category_preds = category_model.predict(X_test_category)
            # Accuracy Scores
            print(f"Category Accuracy: {accuracy_score(y_test_category, category_preds):.2f}")
            logging.info(f"Category Model loading, training and testing successful. Model accuracy {accuracy_score(y_test_category, category_preds):.2f}")

            save_object(self.model_category_save_path, category_model)
            logging.info("Category Model saving successful.")
        except Exception as e:
            CustomException(e, sys)

    def model_trainer_intent(self, X_train_intent, X_test_intent, y_train_intent, y_test_intent):
        try:  
            logging.info("Initializing Intent model loading, fitting and saving.")  
            intent_model = RandomForestClassifier(class_weight='balanced', random_state=42)
            intent_model.fit(X_train_intent, y_train_intent)
            intent_preds = intent_model.predict(X_test_intent)
            # Accuracy Scores
            print(f"Intent Accuracy: {accuracy_score(y_test_intent, intent_preds):.2f}")
            logging.info(f"Intent Model loading, training and testing successful. Model accuracy {accuracy_score(y_test_intent, intent_preds):.2f}")
           
            save_object(self.model_intent_save_path, intent_model)
            logging.info("Intent Model saving successful.")
        except Exception as e:
            CustomException(e, sys)

#training models for Category and Intent prediction
if __name__ == "__main__":

    data_ingestion = DataIngestion()
    model_trainer = ModelTrainer()

    #Data Ingestion
    df = data_ingestion.initiate_data_ingestion()

    # Data Transformation
    data_transformation = DataTransformation(df)

    # training & testing data and model training & saving - category
    X_train_category, X_test_category, y_train_category, y_test_category = data_transformation.output_data_category()
    model_trainer.model_trainer_category(X_train_category, X_test_category, y_train_category, y_test_category)

    # training & testing data and model training & saving - intent
    X_train_intent, X_test_intent, y_train_intent, y_test_intent = data_transformation.output_data_intent()
    model_trainer.model_trainer_intent(X_train_intent, X_test_intent, y_train_intent, y_test_intent)
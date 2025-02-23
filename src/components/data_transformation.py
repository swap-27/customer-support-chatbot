from src.exception import CustomException
import sys
import os
import pandas as pd
from utils import clean_text, save_object
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logging

class DataTransformation:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.preprocessor_category_obj_file_path = os.path.join('artifacts','vectorizer_category.pkl')
        self.preprocessor_intent_obj_file_path = os.path.join('artifacts','vectorizer_intent.pkl')

    def output_data_category(self):
        try:
            logging.info("Initializing Test Train Split for Category and Instruction data")
            X_train, X_test, y_train_category, y_test_category = train_test_split(
                self.df["instruction"], self.df["category"], 
                test_size=0.2, random_state=42, stratify=self.df["category"]
            )

            logging.info("Test Train Split for Category data successful. Initializing vectorization of X_train and X_test without Data Leakage.")
            vectorizer_category = TfidfVectorizer(max_features=5000)
            X_train = vectorizer_category.fit_transform(X_train)  # Fit only on training data
            X_test = vectorizer_category.transform(X_test)  # Transform test data
            save_object(self.preprocessor_category_obj_file_path, vectorizer_category)
            logging.info("Vectorization and saving of Category Vectorizer object successful.")
            return X_train, X_test, y_train_category, y_test_category
        except Exception as e:
            CustomException(e, sys)


    def output_data_intent(self):
        try:
            logging.info("Initializing Test Train Split for Intent and Instruction data")
            X_train, X_test, y_train_intent, y_test_intent = train_test_split(
                self.df["instruction"], self.df["intent"], 
                test_size=0.2, random_state=42, stratify=self.df["intent"]
            )

            logging.info("Test Train Split for Intent data successful. Initializing vectorization of X_train and X_test without Data Leakage.")
            vectorizer_intent = TfidfVectorizer(max_features=5000)
            X_train = vectorizer_intent.fit_transform(X_train)  # Fit only on training data
            X_test = vectorizer_intent.transform(X_test)  # Transform test data
            save_object(self.preprocessor_intent_obj_file_path, vectorizer_intent)
            logging.info("Vectorization and saving of Intent Vectorizer object successful.")
            return X_train, X_test, y_train_intent, y_test_intent
        except Exception as e:
            CustomException(e, sys)

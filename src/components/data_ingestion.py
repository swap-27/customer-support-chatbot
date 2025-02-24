import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException

class DataIngestion:
    def __init__(self):
        self.file_path = os.path.join("artifacts", "customer_support_data.csv")

    def initiate_data_ingestion(self):

        logging.info("Entered the Data Ingestion Method")

        try:

            df = pd.read_csv(self.file_path)
            logging.info(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
            
            return df

        except Exception as e:
            CustomException(e, sys)


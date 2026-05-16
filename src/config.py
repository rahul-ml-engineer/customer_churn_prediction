from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

Raw_Data_Dir = BASE_DIR/"Data"/"raw"
Processed_Data_Dir = BASE_DIR/"Data"/"processed"

Train_Data = Raw_Data_Dir/"churn-bigml-80.csv"
Test_Data = Raw_Data_Dir/"churn-bigml-20.csv"

Processed_Train_Data = Processed_Data_Dir/"Procesed_train.csv"
Feature_eng_Data = Processed_Data_Dir/"Feature_eng_Data.csv"

MODEL_DIR = BASE_DIR/"models"
Model_Path = MODEL_DIR/"Customer_Churn_Prediction.pkl"
Model_Comparison_Path = MODEL_DIR/"Customer_Churn_Model_Comparision.csv"
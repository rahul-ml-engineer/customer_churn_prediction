from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import pandas as pd
from config import Feature_eng_Data, Model_Comparison_Path


def load_data():
    df = pd.read_csv(Feature_eng_Data)
    return df

def split_data(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test

def build_preprocessor(X_train):
    cat_cols = X_train.select_dtypes(include=["object","string"]).columns
    num_cols = X_train.select_dtypes(exclude=["object","string"]).columns
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder(drop="first",handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", numeric_transformer, num_cols),("cat", categorical_transformer, cat_cols)])
    return preprocessor

def get_models():
    models = {"Logistic" : LogisticRegression(max_iter=1000), "DecisionTree" : DecisionTreeClassifier(random_state=42), "RandomForest" : RandomForestClassifier(random_state=42), "KNN" : KNeighborsClassifier(), "XGBoost" : XGBClassifier(random_state=42, eval_metric="logloss")}
    return models

def model_comparison(X_train, y_train, preprocessor): 
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy":"accuracy", "precision":"precision", "recall":"recall", "f1":"f1", "roc_auc":"roc_auc"}
    results=[]
    models=get_models()
    for name,model in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        scores = cross_validate(pipeline, X_train, y_train, cv=skf, scoring=scoring)
        row = {"Model" : name}
        
        for metric in scoring:
            row[metric]=round(scores[f"test_{metric}"].mean(),4)
        
        results.append(row)
    results_df = pd.DataFrame(results)   
    results_df = results_df.sort_values(by="roc_auc", ascending=False)
    return results_df

def save_results(results):
    Model_Comparison_Path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(Model_Comparison_Path, index=False)
    print("Results Saved")

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = build_preprocessor(X_train)
    results=model_comparison(X_train, y_train, preprocessor)
    print(results)
    save_results(results)

if __name__ == "__main__":
    main()

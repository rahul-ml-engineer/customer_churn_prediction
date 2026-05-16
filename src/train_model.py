from config import Feature_eng_Data, Model_Comparison_Path, Model_Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

def load_data():
    df = pd.read_csv(Feature_eng_Data)
    return df

def split_data(df):
    X = df.drop("Churn",axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test

def Preprocessing_Data(X_train):
    cat_cols = X_train.select_dtypes(include=["object","string"]).columns
    num_cols = X_train.select_dtypes(exclude=["object","string"]).columns
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder(drop="first",handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", numeric_transformer, num_cols),("cat", categorical_transformer, cat_cols)])
    return preprocessor

def get_best_model():
    comparison = pd.read_csv(Model_Comparison_Path)
    best = comparison.iloc[0]["Model"]
    print(f"Best model:{best}")
    return best

def hyperparameter_tuning(X_train, y_train, preprocessor):
    model = XGBClassifier(random_state=42, eval_metric="logloss")
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    param_grid={"model__n_estimators" : [100, 200, 300], "model__max_depth" : [3, 5, 7], "model__learning_rate" : [0.01,0.1]}
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    print(grid.best_score_)
    return grid.best_estimator_

def save_model(model):
    Model_Path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Model_Path)
    print("Model Saved")

def main():
    df=load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = Preprocessing_Data(X_train)
    get_best_model()
    final_model = hyperparameter_tuning(X_train, y_train, preprocessor)
    save_model(final_model)

if __name__=="__main__":
    main()
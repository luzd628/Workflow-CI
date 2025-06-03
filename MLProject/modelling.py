import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np 
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # path dataset
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "loan_data_preprocessing.csv")

    data_loan = pd.read_csv(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        data_loan.drop("loan_status", axis=1),
        data_loan["loan_status"],
        random_state=42,
        test_size=0.2
    )

    input_example = X_train[0:5]

    # parameter
    c = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    kernel = sys.argv[2] if len(sys.argv) > 2 else "linear"

    with mlflow.start_run():

        # Train Model
        model = SVC(C=c, kernel=kernel)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        
        # matrik evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log Parameter
        mlflow.log_param("C", c)
        mlflow.log_param("Kernel",kernel)

        # Evaluate Model
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1 score", f1)

        mlflow.sklearn.log_model(
            sk_model = model,
            artifact_path="model",
            input_example=input_example
        )
    
    # simpan model
    path = "./saved_models/svm_model.pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path) 
    print(f"Model telah tersimpan di:{path}")
        
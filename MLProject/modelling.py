import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np 
import warnings
import pickle
from sklearn.model_selection import train_test_split
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

        mlflow.autolog()

        # Train Model
        model = SVC(C=c, kernel=kernel)
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(
            sk_model = model,
            artifact_path="model",
            input_example=input_example
        )

        # Log metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

    # simpan model
    os.makedirs("saved_model",exist_ok=True)
    model_path = os.path.join("saved_model","svm_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model telah tersimpan di:{model_path}")
        
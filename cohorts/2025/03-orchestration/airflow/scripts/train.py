import os
import pickle
import click
import mlflow

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-trip-duration")
mlflow.autolog()


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    with mlflow.start_run():
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        model = LinearRegression()
        model.fit(X_train, y_train)
        print(f"Intercept: {model.intercept_}")

        y_pred = model.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("train_rmse", rmse)
        # Evaluate model on the validation and test sets
        val_rmse = root_mean_squared_error(y_val, model.predict(X_val))
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = root_mean_squared_error(y_test, model.predict(X_test))
        mlflow.log_metric("test_rmse", test_rmse)


if __name__ == '__main__':
    run_train()

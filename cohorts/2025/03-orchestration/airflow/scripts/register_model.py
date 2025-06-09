import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

EXPERIMENT_NAME = "nyc-taxi-trip-duration"
TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "linear-regression-nyc-taxi-trip-duration"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"]
    )
    if not runs:
        print("No runs found in the experiment.")
        return
    best_run = runs[0]
    print(f"Best run ID: {best_run.info.run_id}")
    print(f"Best run test RMSE: {best_run.data.metrics['test_rmse']}")
    print(f"Best run parameters: {best_run.data.params}")

    # Register the best model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)



if __name__ == '__main__':
    run_register_model()

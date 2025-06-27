import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional, Dict, Tuple, Any
import os

def create_mlflow_client_and_run(
    uri: str,
    experiment_name: str,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
    autolog: bool = True,
    user: Optional[str] = None,
    password: Optional[str] = None
) -> Tuple[MlflowClient, Any, str, str]:
    """
    Create an MLflow client, ensure the experiment exists, and start a new run.

    Args:
        uri (str): MLflow tracking server URI (e.g., "http://localhost:5000").
        experiment_name (str): Name of the MLflow experiment to use or create.
        run_name (str, optional): Name for the MLflow run. If None, uses experiment_name.
        tags (dict, optional): Dictionary of tags to set for the run (e.g., metadata, parameters).
        autolog (bool, optional): Whether to enable MLflow autologging. Defaults to True.
        user (str, optional): Username for MLflow server authentication.
        password (str, optional): Password for MLflow server authentication.

    Returns:
        tuple: (mlflow_client, experiment, experiment_id, run_id)
            - mlflow_client (MlflowClient): The MLflow client object.
            - experiment (mlflow.entities.Experiment): The MLflow experiment object.
            - experiment_id (str): The experiment's unique ID.
            - run_id (str): The unique ID of the newly created run.
    """

    mlflow.set_tracking_uri(uri)

    # Set authentication if user and password are provided
    if user is not None and password is not None:
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    if autolog:
        mlflow.autolog()
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
        experiment = client.get_experiment(experiment_id)
    else:
        experiment_id = experiment.experiment_id

    run = client.create_run(
        experiment_id=experiment_id,
        tags={"mlflow.runName": run_name or experiment_name}
    )
    run_id = run.info.run_id

    # Set additional custom tags if provided
    if tags is not None:
        for key, value in tags.items():
            client.set_tag(run_id, key, value)

    return client, experiment, experiment_id, run_id

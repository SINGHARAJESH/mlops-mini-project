import json
import mlflow
import logging
import os
import dagshub


dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "rajeshai2000"
repo_name = "mlops-mini-project"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path:str) -> dict:

    "load the model info from a json file"

    try:
        with open(file_path,'r') as file:
            model_info = json.load(file)

        logger.debug('model loaded from %s',file_path)

        return model_info
    except FileNotFoundError:
        logger.error('file not found',file_path)
        raise
    except Exception as e:
        logger.error('unexcepted error occurred while loading the model info: %s',e)
        raise


def register_model(model_name: str,model_info:dict):

    # register the model to the mlflow model registry

    try:
        model_url = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # register the model

        model_version = mlflow.register_model(model_url,model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
            archive_existing_versions=False
        )
        
        logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():

    try:
        model_info_path = 'reports/experiment_info.json'

        model_info = load_model_info(model_info_path)

        model_name = 'my_model'
        register_model(model_name,model_info)

    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
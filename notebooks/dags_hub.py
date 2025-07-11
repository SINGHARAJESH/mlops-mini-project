import dagshub
import mlflow
mlflow.set_tracking_uri("https://dagshub.com/rajeshai2000/mlops-mini-project.mlflow")
dagshub.init(repo_owner='rajeshai2000', repo_name='mlops-mini-project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
  #https://dagshub.com/rajeshai2000/mlops-mini-project.mlflow/
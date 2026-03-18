'''
Dual experiment logger: logs to both W&B and MLflow simultaneously.
All evaluation modules receive a DualLogger instance and call logger.log() —
they have no knowledge of W&B or MLflow directly.
'''

import mlflow
import wandb


class DualLogger:
    def __init__(self, run_name: str, config: dict, project: str = 'ai-image-eval'):
        self.run_name = run_name
        wandb.init(project=project, name=run_name, config=config)
        mlflow.set_experiment(project)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(config)

    def log(self, metrics: dict, step: int = None):
        wandb.log(metrics, step=step)
        mlflow.log_metrics(metrics, step=step)

    def finish(self):
        wandb.finish()
        mlflow.end_run()

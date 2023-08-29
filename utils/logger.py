from comet_ml import Experiment
from typing import Dict


class CometLogger:
    def __init__(self, run, path_to_config) -> None:
        self.experiment = Experiment(
            auto_output_logging='default',
            auto_metric_logging=True,
            auto_param_logging=True,
            auto_histogram_weight_logging=True,
            auto_histogram_gradient_logging=False,
            auto_histogram_activation_logging=False,
            log_env_details=True,
            log_env_gpu=True,
            api_key="RO93KzhKY0GcxdzCQQg1eyEC1",
            project_name="vader-comparison",
            workspace="imsdcomet")

        self.experiment.set_name(run)
        self.experiment.log_code(file_name=path_to_config)
            
    
    def train(self) -> None:
        self.experiment.train()
    
    def test(self) -> None:
        self.experiment.train()
        
    def log_params(self, params: Dict) -> None:
        self.experiment.log_parameters(parameters=params)
    
    def log_metrics(self,metric: Dict, epoch: int) -> None:
        self.experiment.log_metrics(metric, step=epoch)
        
    def log_model(self, name):
        self.experiment.log_model(name)
    
    def end(self):
        self.experiment.end()
        
    def end(self):
        # self.experiment._on_end(wait=False)
        self.experiment.end()
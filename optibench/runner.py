import yaml
import torch
import numpy as np
import random
import os
from .problems import QuadraticProblem
from .tracker import Tracker

class BenchmarkRunner:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = self.config.get('device', 'cpu')
        self.set_seed(self.config.get('seed', 42))
        self.tracker = Tracker()

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _init_problem(self):
        p_cfg = self.config['problem']
        if p_cfg['name'] == 'quadratic':
            return QuadraticProblem(
                dimensions=p_cfg['dimensions'],
                condition_number=p_cfg['condition_number'],
                device=self.device
            )
        raise ValueError(f"Unknown problem: {p_cfg['name']}")

    def run(self):
        problem = self._init_problem()
        out_dir = self.config['output']['save_dir']
        os.makedirs(out_dir, exist_ok=True)

        for opt_cfg in self.config['optimizers']:
            print(f"Running {opt_cfg['name']}...")
            self.set_seed(self.config.get('seed', 42)) # Ресет сида для честности
            
            # Инициализация параметров модели
            theta = torch.tensor([5.0] * problem.dimensions, 
                                 device=self.device, requires_grad=True)
            
            # Подключение нужного оптимизатора из PyTorch
            if opt_cfg['name'] == 'SGD':
                optimizer = torch.optim.SGD([theta], **opt_cfg['params'])
            elif opt_cfg['name'] == 'Adam':
                optimizer = torch.optim.Adam([theta], **opt_cfg['params'])
            else:
                continue

            iterations = self.config['training']['iterations']
            log_int = self.config['training']['log_interval']

            self.tracker.start_timer()
            for i in range(iterations):
                optimizer.zero_grad()
                loss = problem.compute_loss(theta)
                loss.backward()
                optimizer.step()

                if i % log_int == 0 or i == iterations - 1:
                    grad_norm = torch.norm(theta.grad).item() if theta.grad is not None else 0.0
                    self.tracker.log_step(
                        optimizer_name=opt_cfg['name'],
                        iteration=i,
                        loss=loss.item(),
                        grad_norm=grad_norm,
                        theta=theta.detach().cpu().numpy()
                    )

        df = self.tracker.get_dataframe()
        df.to_csv(os.path.join(out_dir, 'results.csv'), index=False)
        return df

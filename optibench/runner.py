import yaml
import torch
import numpy as np
import random
import os
from .problems import QuadraticProblem, RosenbrockProblem, RastriginProblem
from .tracker import Tracker

class BenchmarkRunner:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.device = self.config.get('device', 'cpu')
        self.set_seed(self.config.get('seed', 42))
        self.tracker = Tracker()
        self.problem = self._init_problem() # Сохраняем объект задачи

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _init_problem(self):
        p_cfg = self.config['problem']
        if p_cfg['name'] == 'quadratic':
            return QuadraticProblem(dimensions=p_cfg.get('dimensions', 2), device=self.device)
        elif p_cfg['name'] == 'rosenbrock':
            return RosenbrockProblem(device=self.device)
        elif p_cfg['name'] == 'rastrigin':
            return RastriginProblem(dimensions=p_cfg.get('dimensions', 2), device=self.device)
        raise ValueError(f"Unknown problem: {p_cfg['name']}")

    def run(self):
        out_dir = self.config['output']['save_dir']
        os.makedirs(out_dir, exist_ok=True)
        init_coords = self.config['problem'].get('init_theta', [5.0] * self.problem.dimensions)

        for opt_cfg in self.config['optimizers']:
            print(f"Running {opt_cfg['name']}...")
            self.set_seed(self.config.get('seed', 42))
            
            theta = torch.tensor(init_coords, device=self.device, requires_grad=True)
            
            # Зоопарк оптимизаторов
            opt_class = getattr(torch.optim, opt_cfg['name'], None)
            if opt_class is None:
                print(f"Optimizer {opt_cfg['name']} not found in torch.optim!")
                continue
            optimizer = opt_class([theta], **opt_cfg['params'])

            iterations = self.config['training']['iterations']
            log_int = self.config['training']['log_interval']

            self.tracker.start_timer()
            for i in range(iterations):
                optimizer.zero_grad()
                loss = self.problem.compute_loss(theta)
                loss.backward()
                optimizer.step()

                if i % log_int == 0 or i == iterations - 1:
                    grad_norm = torch.norm(theta.grad).item() if theta.grad is not None else 0.0
                    self.tracker.log_step(opt_cfg['name'], i, loss.item(), grad_norm, theta.detach().cpu().numpy())

        df = self.tracker.get_dataframe()
        df.to_csv(os.path.join(out_dir, 'results.csv'), index=False)
        return df

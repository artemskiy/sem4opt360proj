import yaml
import torch
import numpy as np
import random
import os
from .problems import QuadraticProblem, RosenbrockProblem, RastriginProblem, MNISTProblem
from .tracker import Tracker

class BenchmarkRunner:
    def __init__(self, config_dict: dict):
        # Теперь принимаем словарь, чтобы было удобно делать Grid Search из кода
        self.config = config_dict
        self.device = self.config.get('device', 'cpu')
        self.tracker = Tracker()
        self.set_seed(self.config.get('seed', 42))
        self.problem = self._init_problem()

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
        elif p_cfg['name'] == 'mnist':
            return MNISTProblem(batch_size=p_cfg.get('batch_size', 128), device=self.device)
        raise ValueError(f"Unknown problem: {p_cfg['name']}")

    def run(self):
        out_dir = self.config['output']['save_dir']
        os.makedirs(out_dir, exist_ok=True)
        
        is_nn = isinstance(self.problem, MNISTProblem)

        for opt_cfg in self.config['optimizers']:
            print(f"Running {opt_cfg['name']} (Params: {opt_cfg['params']})...")
            self.set_seed(self.config.get('seed', 42))
            
            if is_nn:
                self.problem = self._init_problem() # Реинициализируем веса модели для честности
                params = self.problem.get_parameters()
                theta = [] # Для НС траектории не пишем
            else:
                init_coords = self.config['problem'].get('init_theta', [5.0] * self.problem.dimensions)
                theta = torch.tensor(init_coords, device=self.device, requires_grad=True)
                params = [theta]

            opt_class = getattr(torch.optim, opt_cfg['name'])
            optimizer = opt_class(params, **opt_cfg['params'])

            iterations = self.config['training']['iterations']
            log_int = self.config['training']['log_interval']

            self.tracker.start_timer()
            for i in range(iterations):
                optimizer.zero_grad()
                loss = self.problem.compute_loss(theta if not is_nn else None)
                loss.backward()
                optimizer.step()

                if i % log_int == 0 or i == iterations - 1:
                    # Норму градиента для всей НС считать долго, пропускаем для простоты
                    grad_norm = torch.norm(theta.grad).item() if not is_nn and theta.grad is not None else 0.0
                    opt_label = f"{opt_cfg['name']}_{opt_cfg['params'].get('lr', 'X')}"
                    self.tracker.log_step(opt_label, i, loss.item(), grad_norm, theta.detach().cpu().numpy() if not is_nn else [])

        df = self.tracker.get_dataframe()
        df.to_csv(os.path.join(out_dir, 'results.csv'), index=False)
        return df

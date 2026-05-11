import torch

class BaseProblem:
    def __init__(self, device='cpu'):
        self.device = device

    def compute_loss(self, theta):
        raise NotImplementedError

class QuadraticProblem(BaseProblem):
    def __init__(self, dimensions=2, condition_number=100.0, device='cpu'):
        super().__init__(device)
        self.dimensions = dimensions
        
        # Генерируем матрицу A с заданным числом обусловленности
        # В простейшем случае диагональная матрица, где lambda_max / lambda_min = condition_number
        diag = torch.linspace(1.0, condition_number, dimensions, device=device)
        self.A = torch.diag(diag)

    def compute_loss(self, theta):
        # f(x) = 0.5 * x^T * A * x
        return 0.5 * torch.dot(theta, torch.mv(self.A, theta))

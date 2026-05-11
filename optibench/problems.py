import torch
import math

class BaseProblem:
    def __init__(self, device='cpu'):
        self.device = device

    def compute_loss(self, theta):
        raise NotImplementedError

class QuadraticProblem(BaseProblem):
    def __init__(self, dimensions=2, condition_number=100.0, device='cpu'):
        super().__init__(device)
        self.dimensions = dimensions
        diag = torch.linspace(1.0, condition_number, dimensions, device=device)
        self.A = torch.diag(diag)

    def compute_loss(self, theta):
        return 0.5 * torch.dot(theta, torch.mv(self.A, theta))

class RosenbrockProblem(BaseProblem):
    def __init__(self, a=1.0, b=100.0, device='cpu'):
        super().__init__(device)
        self.a = a
        self.b = b
        self.dimensions = 2

    def compute_loss(self, theta):
        # f(x, y) = (a - x)^2 + b(y - x^2)^2
        x, y = theta[0], theta[1]
        return (self.a - x)**2 + self.b * (y - x**2)**2

class RastriginProblem(BaseProblem):
    def __init__(self, dimensions=2, A=10.0, device='cpu'):
        super().__init__(device)
        self.dimensions = dimensions
        self.A = A

    def compute_loss(self, theta):
        # f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
        return self.A * self.dimensions + torch.sum(theta**2 - self.A * torch.cos(2 * math.pi * theta))

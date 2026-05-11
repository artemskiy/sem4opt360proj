import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math

class BaseProblem:
    def __init__(self, device='cpu'):
        self.device = device

    def compute_loss(self, theta=None):
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
        self.dimensions = 2
        self.a, self.b = a, b

    def compute_loss(self, theta):
        x, y = theta[0], theta[1]
        return (self.a - x)**2 + self.b * (y - x**2)**2

class RastriginProblem(BaseProblem):
    def __init__(self, dimensions=2, A=10.0, device='cpu'):
        super().__init__(device)
        self.dimensions = dimensions
        self.A = A

    def compute_loss(self, theta):
        return self.A * self.dimensions + torch.sum(theta**2 - self.A * torch.cos(2 * math.pi * theta))

class MNISTProblem(BaseProblem):
    def __init__(self, batch_size=128, device='cpu'):
        super().__init__(device)
        self.dimensions = -1 # Сигнал, что мы не строим 2D графики
        
        # Двухслойный перцептрон (MLP)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ).to(device)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Скачиваем датасет (сохранится в папку ./data)
        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.data_iter = iter(self.dataloader)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self):
        return self.model.parameters()

    def compute_loss(self, theta=None):
        try:
            data, target = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            data, target = next(self.data_iter)
            
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        loss = self.criterion(output, target)
        return loss

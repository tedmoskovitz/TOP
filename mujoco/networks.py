import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLPNetwork(nn.Module):
    
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 256) -> None:
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
                        nn.Linear(input_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_dim),
                        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Policy(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256) -> None:
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.network = MLPNetwork(state_dim, action_dim, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = self.tanh(x)
        return x


class DoubleQFunc(nn.Module):
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256) -> None:
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)

class QFunc(nn.Module):
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256) -> None:
        super(QFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((state, action), dim=1)
        return self.network1(x)

class QuantileDoubleQFunc(nn.Module):
    
    def __init__(self, state_dim: int, action_dim: int, n_quantiles: int = 100, hidden_size: int = 256) -> None:
        super(QuantileDoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, n_quantiles, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, n_quantiles, hidden_size)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)

class QuantileQFunc(nn.Module):
    
    def __init__(self, state_dim: int, action_dim: int, n_quantiles: int = 100, hidden_size: int = 256) -> None:
        super(QuantileQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, n_quantiles, hidden_size)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((state, action), dim=1)
        return self.network1(x)

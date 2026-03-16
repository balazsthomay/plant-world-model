from dataclasses import dataclass

import torch
import torch.nn as nn

from src.configs import EnsembleConfig

MIN_LOG_VAR = -10.0
MAX_LOG_VAR = 0.5


@dataclass
class EnsemblePrediction:
    """Prediction from the dynamics ensemble."""

    mean: torch.Tensor
    epistemic_var: torch.Tensor
    aleatoric_var: torch.Tensor


class ProbabilisticMLP(nn.Module):
    """MLP that outputs a Gaussian distribution (mean + log_variance)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: list[int],
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        act_fn = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_fn())
            prev_dim = h

        self.trunk = nn.Sequential(*layers)
        # Output: mean and log_variance for each output dimension
        self.head = nn.Linear(prev_dim, output_dim * 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        out = self.head(features)
        mean = out[:, : self.output_dim]
        log_var = out[:, self.output_dim :]
        log_var = torch.clamp(log_var, MIN_LOG_VAR, MAX_LOG_VAR)
        return mean, log_var

    def loss(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Gaussian negative log-likelihood loss."""
        mean, log_var = self(x)
        var = torch.exp(log_var)
        # NLL = 0.5 * (log(var) + (target - mean)^2 / var)
        nll = 0.5 * (log_var + (targets - mean) ** 2 / var)
        return nll.mean()


class DynamicsEnsemble:
    """Ensemble of probabilistic MLPs for dynamics prediction."""

    def __init__(
        self, config: EnsembleConfig, state_dim: int, action_dim: int
    ) -> None:
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        input_dim = state_dim + action_dim
        output_dim = state_dim

        self.networks = nn.ModuleList(
            [
                ProbabilisticMLP(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_sizes=config.hidden_sizes,
                    activation=config.activation,
                )
                for _ in range(config.n_networks)
            ]
        )

    def predict(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> EnsemblePrediction:
        """Predict next-state deltas with uncertainty estimates.

        Returns mean prediction, epistemic uncertainty (variance of means),
        and aleatoric uncertainty (mean of variances).
        """
        x = torch.cat([states, actions], dim=-1)

        all_means = []
        all_vars = []
        for net in self.networks:
            with torch.no_grad():
                mean, log_var = net(x)
            all_means.append(mean)
            all_vars.append(torch.exp(log_var))

        means_stack = torch.stack(all_means)  # (n_networks, batch, output_dim)
        vars_stack = torch.stack(all_vars)

        ensemble_mean = means_stack.mean(dim=0)
        epistemic_var = means_stack.var(dim=0)
        aleatoric_var = vars_stack.mean(dim=0)

        return EnsemblePrediction(
            mean=ensemble_mean,
            epistemic_var=epistemic_var,
            aleatoric_var=aleatoric_var,
        )

    def sample_prediction(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """TS-1 sampling: pick a random ensemble member for each sample."""
        x = torch.cat([states, actions], dim=-1)
        batch_size = x.shape[0]
        n = len(self.networks)

        # Randomly assign each sample to a network
        indices = torch.randint(0, n, (batch_size,))
        result = torch.zeros(batch_size, self.state_dim)

        for i, net in enumerate(self.networks):
            mask = indices == i
            if mask.any():
                with torch.no_grad():
                    mean, log_var = net(x[mask])
                std = torch.exp(0.5 * log_var)
                result[mask] = mean + std * torch.randn_like(std)

        return result

    def to(self, device: torch.device) -> "DynamicsEnsemble":
        self.networks = self.networks.to(device)
        return self

    def save(self, path: str) -> None:
        torch.save(
            {
                "networks": self.networks.state_dict(),
                "config": self.config,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "DynamicsEnsemble":
        data = torch.load(path, weights_only=False)
        ensemble = cls(
            config=data["config"],
            state_dim=data["state_dim"],
            action_dim=data["action_dim"],
        )
        ensemble.networks.load_state_dict(data["networks"])
        return ensemble

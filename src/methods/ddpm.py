"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

import math
from typing import Dict, Tuple, Optional, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        # TODO: Add your own arguments here
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self. beta_start = beta_start
        self. beta_end = beta_end

        betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.register_buffer("sqrt_recip_alphas", (1.0 / alphas).sqrt())
        self.register_buffer("posterior_variance", betas)   # σ_t² = β_t

    # =========================================================================
    # You can add, delete or modify as many functions as you would like
    # =========================================================================
    
    # Pro tips: If you have a lot of pseudo parameters that you will specify for each
    # model run but will be fixed once you specified them (say in your config),
    # then you can use super().register_buffer(...) for these parameters

    # Pro tips 2: If you need a specific broadcasting for your tensors,
    # it's a good idea to write a general helper function for that

    # Important! Must Understand!
    def _extract(self, coef: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return coef[t].view(-1, *([1] * (x.dim() - 1)))
    # =========================================================================
    # Forward process
    # =========================================================================

    def forward_process(self, x_0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        if eps is None:
            eps = torch.randn_like(x_0)
        a = self._extract(self.sqrt_alphas_cumprod, t, x_0)
        b = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0)
        return a * x_0 + b * eps
    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width)
            **kwargs: Additional method-specific arguments
        
        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging (e.g., {'mse': 0.1})
        """
        B = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=x_0.device)
        eps = torch.randn_like(x_0)
        x_t = self.forward_process(x_0, t, eps)  # compute loss includes a forward pass first
        eps_pred = self.model(x_t, t)  # predict noise
        loss = F.mse_loss(eps_pred, eps)
        return loss, {"mse": loss.item()}

    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================
    
    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: the time
            **kwargs: Additional method-specific arguments
        
        Returns:
            x_prev: Noisy samples at time t-1 (batch_size, channels, height, width)
        """
        eps_pred = self.model(x_t, t)
        # beta / sqrt(1 - alpha_cumprod)
        coef = self._extract(self.betas, t, x_t / self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t))
        # 1/sqrt(alpha) * (x_t - beta / sqrt(1 - alpha_cumprod) * eps_pred)
        mean = self._extract(self.sqrt_recip_alphas, t, x_t) * (x_t - coef * eps_pred)

        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(-1, *([1] * (x_t.dim() - 1)))  # no noise when t == 0
        sigma = self._extract(self.posterior_variance, t, x_t).sqrt()
        return mean + nonzero_mask * sigma * noise

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        **kwargs
    ) -> torch.Tensor:
        """
        TODO: Implement DDPM sampling loop: start from pure noise, iterate through all the time steps using reverse_process()

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            **kwargs: Additional method-specific arguments (e.g., num_steps)
        
        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.eval_mode()
        x_t = torch.randn(
            batch_size,
            *image_shape,
            device=self.device
        )

        for step in reversed(range(self.num_timesteps)):
            t = torch.full(
                (batch_size,),
                step,
                device=self.device,
                dytpe = torch.long
            )
            x_t = self.reverse_process(x_t, t)

        return x_t
    # =========================================================================
    # Device / state
    # =========================================================================

    def to(self, device: torch.device) -> "DDPM":
        super().to(device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        # TODO: add other things you want to save
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
            # TODO: add your parameters here
        )

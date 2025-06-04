import torch
import numpy as np


def eps_from_v(z_0, z_t, sigma_t):
    return (z_t - z_0) / sigma_t

def v_to_eps(v, t, x_t):
    """
    function to compute the epsilon parametrization from the velocity field
    with x_t = t * x_0 + (1 - t) * x_1 with x_0 ~ N(0,I)
    """
    eps_t = (1-t)*v + x_t
    return eps_t


def clip_gradients(gradients, clip_value):
    grad_norm = gradients.norm(dim=2)
    mask = grad_norm > clip_value
    mask_exp = mask[:, :, None].expand_as(gradients)
    gradients[mask_exp] = (
        gradients[mask_exp]
        / grad_norm[:, :, None].expand_as(gradients)[mask_exp]
        * clip_value
    )
    return gradients


class Adam:

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = torch.zeros_like(parameters)
        self.v = torch.zeros_like(parameters)

    def step(self, params, grad) -> torch.Tensor:
        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # check if self.lr is callable
        if callable(self.lr):
            lr = self.lr(self.t - 1)
        else:
            lr = self.lr
        update = lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

        return params - update


def make_cosine_decay_schedule(
    init_value: float,
    total_steps: int,
    alpha: float = 0.0,
    exponent: float = 1.0,
    warmup_steps=0,
):
    def schedule(count):
        if count < warmup_steps:
            # linear up
            return (init_value / warmup_steps) * count
        else:
            # half cosine down
            decay_steps = total_steps - warmup_steps
            count = min(count - warmup_steps, decay_steps)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * count / decay_steps))
            decayed = (1 - alpha) * cosine_decay**exponent + alpha
            return init_value * decayed

    return schedule


def make_linear_decay_schedule(
    init_value: float, total_steps: int, final_value: float = 0, warmup_steps=0
):
    def schedule(count):
        if count < warmup_steps:
            # linear up
            return (init_value / warmup_steps) * count
        else:
            # linear down
            decay_steps = total_steps - warmup_steps
            count = min(count - warmup_steps, decay_steps)
            return init_value - (init_value - final_value) * count / decay_steps

    return schedule


def clip_norm_(tensor, max_norm):
    norm = tensor.norm()
    if norm > max_norm:
        tensor.mul_(max_norm / norm)


def lr_warmup(step, warmup_steps):
    return min(1.0, step / max(warmup_steps, 1))


def linear_decay_lambda(step, warmup_steps, decay_steps, total_steps):
    if step < warmup_steps:
        min(1.0, step / max(warmup_steps, 1))
    else:
        # linear down
        # decay_steps = total_steps - warmup_steps
        count = min(step - warmup_steps, decay_steps)
        return 1 - count / decay_steps

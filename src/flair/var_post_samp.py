import torch
import numpy as np
import sys
import os
import tqdm
from flair import degradations
import torchvision

def total_variation_loss(x):
    """
    Compute the total variation loss for a batch of images.
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W)
    Returns:
        torch.Tensor: Total variation loss
    """
    # Compute the differences between adjacent pixels
    diff_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    diff_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    
    # Sum the differences
    return torch.sum(diff_x) + torch.sum(diff_y)

class VariationalPosterior:
    def __init__(self, model, config):
        """
        Args:
            model (torch.nn.Module): model to be used for inference, should have single_step, encode and decode methods
            config (dict): configuration

        """
        self.config = config
        self.model = model

        # Initialize degradation model
        try:
            degradation = getattr(degradations, config["degradation"]["name"])
        except AttributeError:
            print(f"Degradation {config['degradation']['name']} not defined.")
            sys.exit(1)

        self.forward_operator = degradation(**config["degradation"]["kwargs"])

        if "optimized_reg_weight" in config and config["optimized_reg_weight"]:
            reg_weight = np.load(config["optimized_reg_weight"])
            if "reg_weight" in config["optimized_reg_weight"]:
                self.regularizer_weight = reg_weight * config["regularizer_weight"]
            else:
                # reg_weight = reg_weight / np.nanmax(reg_weight)
                reg_weight = 1 / (reg_weight + 1e-7)
                reg_weight = reg_weight / np.nansum(reg_weight) * reg_weight.shape[0]
                if "reg-shift" in config:
                    self.regularizer_weight = reg_weight + config["reg-shift"]
                else:
                    self.regularizer_weight = reg_weight - reg_weight[-1]
                self.regularizer_weight = np.clip(self.regularizer_weight, 0, None) * config["regularizer_weight"] 
            print("loaded opt reg weight.")
        else:
            self.regularizer_weight = config["regularizer_weight"]

    def set_degradation(self):
        try:
            degradation = getattr(degradations, self.config["degradation"]["name"])
        except AttributeError:
            sys.exit(1)
        self.forward_operator = degradation(**self.config["degradation"]["kwargs"])

    def data_term(self, latent_mu, y, optimizer_dataterm, likelihood_weight, likelihood_steps, early_stopping):
        """
        Performs data term optimization over several steps with early stopping.
        """
        for k in range(likelihood_steps):
            with torch.enable_grad():
                data_loss = torch.nn.MSELoss(reduction='sum')(self.forward_operator(self.model.decode(latent_mu), noise=False), y)
                loss = likelihood_weight * data_loss.sum()
            if data_loss < early_stopping * y.numel():
                if latent_mu.grad is not None:
                    latent_mu.grad = None
                del loss
                del data_loss
                break
            loss.backward()
            optimizer_dataterm.step()
            optimizer_dataterm.zero_grad()
            del loss
            del data_loss
        return 

    @torch.no_grad()
    def projection(self, latent_mu, y, alpha=1):
        x_0 = self.model.decode(latent_mu)
        y_hat = self.forward_operator(x_0, noise=False)
        x_inv_hat = self.forward_operator.pseudo_inv(y_hat)
        projection = x_0 - x_inv_hat + self.forward_operator.pseudo_inv(y) 
        latent_projection = self.model.encode(projection)
        # soft projection in latent space
        latent_projection = latent_mu - (latent_mu - latent_projection) * alpha
        return projection, latent_projection

    def find_closest_t(self, t):
        ts = torch.linspace(1, 0.0, self.regularizer_weight.shape[0], device=t.device, dtype=t.dtype)
        return torch.argmin(torch.abs(ts - t))

    def forward(self, y, kwargs):
        """
        Uses variational approach to infer the mode of the posterior distribution given a measurement y.
        Args:
            y (torch.Tensor): measurement tensor
        Returns:
            torch.Tensor: estimated mu
        """
        
        for key, value in kwargs.items():
            try:
                kwargs[key] = value.to(y.device)
            except AttributeError:
                pass

        return_dict = {}
        device = y.device

        
        if "init" in self.config and self.config["init"] =="random":
            # TODO: put this in model wrapper
            shape = (
                1,
                16,
                int(self.config["resolution"]) // self.model.vae_scale_factor,
                int(self.config["resolution"]) // self.model.vae_scale_factor,
            )
            latent_mu = torch.randn(shape, device=device, dtype=y.dtype)
        else:
            x_inv = self.forward_operator.pseudo_inv(y)
            latent_mu = self.model.encode(x_inv)
        latent_mu = latent_mu.detach().clone()
        latent_mu.requires_grad = True

        start_noise = torch.randn_like(latent_mu)
        optim_noise = start_noise.detach().clone()
        timesteps = self._get_timesteps(device)

        for epoch in range(self.config["epochs"]):
            optimizer, optimizer_dataterm = self._initialize_optimizers(latent_mu)
    
            for i, t in tqdm.tqdm(enumerate(timesteps), desc="Variational Optimization", total=len(timesteps)):
                t = torch.tensor([t], device=device, dtype=latent_mu.dtype)
                kwargs["noise"] = optim_noise.detach()
                kwargs["inv_alpha"] = self.config["inv_alpha"]
                    

                eps_prediction, noise, a_t, sigma_t, v_pred = self.model.single_step(latent_mu, t, kwargs)

                # predict x1 which is the start noise vector for DTA
                optim_noise = a_t * latent_mu + sigma_t * noise + a_t * v_pred

                reg_term = self._compute_regularization_term(eps_prediction, noise, a_t, sigma_t, t, latent_mu, v_pred)

                if self.config["likelihood_weight_mode"] == "reg_weight":
                    reg_idx = self.find_closest_t(t)
                    likelihood_weight = self.regularizer_weight[reg_idx] * self.config["likelihood_weight"]
                else:
                    likelihood_weight = self.config["likelihood_weight"]

                with torch.enable_grad():
                    reg_term = (reg_term.detach() * latent_mu.view(reg_term.shape[0], -1)).sum()

                    
                reg_term.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                if self.config["projection"] and t>0.7:
                    with torch.enable_grad():
                        _, latent_mu_projection = self.projection(latent_mu, y)
                        proj_loss = (latent_mu - latent_mu_projection).detach() * latent_mu
                        proj_loss = proj_loss.sum()
                    optimizer_dataterm.zero_grad()
                    proj_loss.backward()
                    optimizer_dataterm.step()
                    optimizer_dataterm.zero_grad()
                
                self.data_term(
                    latent_mu,
                    y.detach(),
                    optimizer_dataterm,
                    likelihood_weight,
                    self.config["likelihood_steps"],
                    self.config["early_stopping"]
                )
                # self.save_intermediate_results(latent_mu, i)

        x_hat = self.model.decode(latent_mu)
        return_dict.update({"x_hat": x_hat})
        return return_dict

    def _get_timesteps(self, device):
        timesteps = self.model.get_timesteps(self.config["n_steps"], device=device, ts_min=self.config["ts_min"])
        if self.config["t_sampling"] == "descending":
            return timesteps
        elif self.config["t_sampling"] == "ascending":
            return timesteps.flip(0)
        elif self.config["t_sampling"] == "random":
            idx = torch.randperm(len(timesteps), device=device, dtype=timesteps.dtype)
            return timesteps[idx]            
        else:
            raise ValueError(f't_sampling {self.config["t_sampling"]} not supported.')

    def _initialize_optimizers(self, latent_mu):
        params = [latent_mu]
        params2 = [latent_mu]
        optimizer = self._get_optimizer(self.config["optimizer"], params)
        optimizer_dataterm = self._get_optimizer(self.config["optimizer_dataterm"], params2)
        if "scheduler" in self.config:
            self.scheduler = self._get_scheduler(self.config["scheduler"], optimizer)
        if "scheduler_dataterm" in self.config:
            self.scheduler_dataterm = self._get_scheduler(self.config["scheduler_dataterm"], optimizer_dataterm)
        return optimizer, optimizer_dataterm

    def _get_optimizer(self, optimizer_config, params):
        if optimizer_config["name"] == "Adam":
            return torch.optim.Adam(params, **optimizer_config["kwargs"])
        elif optimizer_config["name"] == "SGD":
            return torch.optim.SGD(params, **optimizer_config["kwargs"])
        else:
            raise ValueError(f'optimizer {optimizer_config["name"]} not supported.')

    def _get_scheduler(self, scheduler_config, optimizer):
        if scheduler_config["name"] == "StepLR":
            return torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_config["kwargs"])
        elif scheduler_config["name"] == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config["kwargs"])
        elif scheduler_config["name"] == "LinearLR":
            return torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_config["kwargs"])
        # Add other schedulers as needed
        else:
            raise ValueError(f'scheduler {scheduler_config["name"]} not supported.')

    def _compute_regularization_term(self, eps_prediction, noise, a_t, sigma_t, t, latent_mu, v):
        reg_term = (eps_prediction - noise).reshape(eps_prediction.shape[0], -1)
        # reg_term = (latent_mu-(a_t*latent_mu + sigma_t*noise - t *v)).reshape(eps_prediction.shape[0], -1)
        # 
        # reg_term /= reg_term.norm() / 1000
        if self.config["lambda_func"] == "sigma2":
            reg_term *= sigma_t / a_t
        elif self.config["lambda_func"] == "v":
            x_t = a_t*latent_mu + sigma_t*noise
            lambda_t_der = -2 * (1/(1-t) + 1/t)
            reg_term = lambda_t_der * t * reg_term / 2
            u_t = - 1 / (1-t) * x_t - t * lambda_t_der / 2 * noise
            # u_t = noise - latent_mu
            reg_term = -(u_t - v).reshape(eps_prediction.shape[0], -1)
        elif self.config["lambda_func"] != "sigma":
            raise ValueError(f'lambda_func {self.config["lambda_func"]} not supported.')

        if isinstance(self.regularizer_weight, np.ndarray):
            reg_idx = self.find_closest_t(t)
            regularizer_weight = self.regularizer_weight[reg_idx]
        else:
            regularizer_weight = self.regularizer_weight

        return reg_term * regularizer_weight
    
    def save_intermediate_results(self, latent_mu, i):
        """
        Saves intermediate results for debugging or visualization.
        Args:
            latent_mu (torch.Tensor): current latent representation
            i (int): current iteration index
        """
        x_hat = self.model.decode(latent_mu)
        # create directory if it does not exist
        os.makedirs("intermediate_results", exist_ok=True)
        torchvision.utils.save_image(x_hat, f"intermediate_results/x_hat_{i}.png", normalize=True, value_range=(-1, 1))
        print(f"Saved intermediate results for iteration {i}.")
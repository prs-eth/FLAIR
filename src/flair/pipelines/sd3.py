import torch
from typing import Dict, Any
from diffusers.pipelines.stable_diffusion_3 import pipeline_stable_diffusion_3
from flair.pipelines import utils
import tqdm

class SD3Wrapper(pipeline_stable_diffusion_3.StableDiffusion3Pipeline):
    def to(self, device, kwargs):
        self.transformer.to(device)
        self.vae.to(device)
        return self

    def get_timesteps(self, n_steps, device, ts_min=0):
        # Create a linear schedule for timesteps
        timesteps = torch.linspace(1, ts_min, n_steps+2, device=device, dtype=torch.float32)
        return timesteps[1:-1]  # Exclude the first and last timesteps

    def single_step(
        self,
        img_latent: torch.Tensor,
        t: torch.Tensor,
        kwargs: Dict[str, Any],
        is_noised_latent = False,
    ):  
        if "noise" in kwargs:
            noise = kwargs["noise"].detach()
            alpha = kwargs["inv_alpha"]
            if alpha == "tsqrt":
                alpha = t**0.5  # * 0.75
            elif alpha == "t":
                alpha = t
            elif alpha == "sine":
                alpha = torch.sin(t * 3.141592653589793/2)
            elif alpha == "1-t":
                alpha = 1 - t
            elif alpha == "1-t*0.5":
                alpha = (1 - t)*0.5
            elif alpha == "1-t*0.9":
                alpha = (1 - t)*0.9
            elif alpha == "t**1/3":
                alpha = t**(1/3)
            elif alpha == "(1-t)**0.5":
                alpha = (1-t)**0.5
            elif alpha == "((1-t)*0.8)**0.5":
                alpha = (1-t*0.8)**0.5
            elif alpha == "(1-t)**2":
                alpha = (1-t)**2
            # alpha = t * kwargs["inv_alpha"]
            noise = (alpha) * noise + (1-alpha**2)**0.5 * torch.randn_like(img_latent)
            # noise = noise / noise.std()
            # noise = noise / (1- 2*alpha*(1-alpha))**0.5
            # noise = noise + alpha * torch.randn_like(img_latent)
        else:
            noise = torch.randn_like(img_latent)
        if is_noised_latent:
            noised_latent = img_latent
        else:
            noised_latent = t * noise + (1 - t) * img_latent
        latent_model_input = torch.cat([noised_latent] * 2) if self.do_classifier_free_guidance else noised_latent
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latent_model_input.shape[0])
        noise_pred = self.transformer(
            hidden_states=latent_model_input.to(img_latent.dtype),
            timestep=(timestep*1000).to(img_latent.dtype),
            encoder_hidden_states=kwargs["prompt_embeds"].repeat(img_latent.shape[0], 1, 1),
            pooled_projections=kwargs["pooled_prompt_embeds"].repeat(img_latent.shape[0], 1),
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
  
        eps = utils.v_to_eps(noise_pred, t, noised_latent)
        return eps, noise, (1 - t), t, noise_pred
    
    def encode(self, img):
        # Encode the image into latent space
        
        img_latent = self.vae.encode(img, return_dict=False)[0]
        if hasattr(img_latent, "sample"):
            img_latent = img_latent.sample()
        img_latent = (img_latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return img_latent

    def decode(self, img_latent):
        # Decode the latent representation back to image space
        img = self.vae.decode(img_latent / self.vae.config.scaling_factor + self.vae.config.shift_factor, return_dict=False)[0]
        return img

    def denoise(self, pseudo_inv, kwargs, inverse=False):
        # get timesteps
        timesteps = torch.linspace(1, 0, kwargs["n_steps"], device=pseudo_inv.device, dtype=pseudo_inv.dtype)
        sigmas = timesteps
        if inverse:
            timesteps = timesteps.flip(0)
            sigmas = sigmas.flip(0)
        
        # make a single step
        for i, t in tqdm.tqdm(enumerate(timesteps[:-1]), desc="Denoising", total=len(timesteps)-1):
            eps, noise, _, t, v = self.single_step(
                pseudo_inv,
                t.to("cuda")*1000,
                kwargs,
                is_noised_latent=True,
            )
            # step
            sigma_next = sigmas[i+1]
            sigma_t = sigmas[i]
            pseudo_inv = pseudo_inv + v * (sigma_next - sigma_t)
        return pseudo_inv
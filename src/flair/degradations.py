import torch
import numpy as np
from munch import munchify
from scipy.ndimage import distance_transform_edt
from flair.functions.degradation import get_degradation
import torchvision 

class BaseDegradation(torch.nn.Module):
    def __init__(self, noise_std=0.0):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x):
        x = x + self.noise_std * torch.randn_like(x)
        return x

    def pseudo_inv(self, y):
        return y


def zero_filler(x, scale):
    B, C, H, W = x.shape
    scale = int(scale)
    H_new, W_new = H * scale, W * scale
    out = torch.zeros(B, C, H_new, W_new, dtype=x.dtype, device=x.device)
    out[:, :, ::scale, ::scale] = x
    return out


class SuperRes(BaseDegradation):
    def __init__(self, scale, noise_std=0.0, img_size=256):
        super().__init__(noise_std=noise_std)
        self.scale = scale
        deg_config = munchify({
        'channels': 3,
        'image_size': img_size,
        'deg_scale': scale
        })
        self.img_size = img_size
        self.deg = get_degradation("sr_bicubic", deg_config, device="cuda")

    def forward(self, x, noise=True):
        dtype = x.dtype
        y = self.deg.A(x.float())
        # add noise
        if noise:
            y = super().forward(y)

        return y.to(dtype)

    def pseudo_inv(self, y):
        x = self.deg.At(y.float()).reshape(1,3,self.img_size, self.img_size)* self.scale**2
        return x.to(y.dtype)
    
    
    def nn(self, y):
        x = torch.nn.functional.interpolate(
            y.reshape(1,3,self.img_size//self.scale, self.img_size//self.scale), scale_factor=self.scale, mode="nearest"
        )
        return x.to(y.dtype)
    
class SuperResGradio(BaseDegradation):
    def __init__(self, scale, noise_std=0.0, img_size=256):
        super().__init__(noise_std=noise_std)
        self.scale = scale
        self.downscaler = lambda x: torch.nn.functional.interpolate(
            x.float(), scale_factor=1/self.scale, mode="bilinear", align_corners=False, antialias=True
        )
        self.upscaler = lambda x: torch.nn.functional.interpolate(
            x.float(), scale_factor=self.scale, mode="bilinear", align_corners=False, antialias=True
        )
        self.img_size = img_size 

    def forward(self, x, noise=True):
        dtype = x.dtype
        y = self.downscaler(x.float())
        # add noise
        if noise:
            y = super().forward(y)

        return y.to(dtype)

    def pseudo_inv(self, y):
        x = self.upscaler(y.float())
        return x.to(y.dtype)
    
    
    def nn(self, y):
        x = torch.nn.functional.interpolate(
            y.reshape(1,3,self.img_size//self.scale, self.img_size//self.scale), scale_factor=self.scale, mode="nearest-exact"
        )
        return x.to(y.dtype)


class Inpainting(BaseDegradation):
    def __init__(self, mask, H, W, noise_std=0.0):
        """
        mask: torch.Tensor, shape (H, W), dtype bool
        function assumes 3 channels
        """
        super().__init__(noise_std=noise_std)
        if isinstance(mask, list):
            # generate box from left, right, lower upper list
            # observed region is True
            mask_ = torch.ones(H, W, dtype=torch.bool)
            mask_[slice(*mask[0:2]), slice(*mask[2:])] = False
            # repeat for 3 channels
            mask_ = mask_.repeat(3, 1, 1)
        elif isinstance(mask, str):
            # load mask file
            mask_ = torch.tensor(np.load(mask), dtype=torch.bool)
            mask_ = mask_.repeat(3, 1, 1)
        elif isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                # assume mask is for one channel, repeat for 3 channels
                mask_ = mask[None].repeat(3, 1, 1)
            elif mask.ndim == 3 and mask.shape[0] == 1:
                # assume mask is for one channel, repeat for 3 channels
                mask_ = mask.repeat(3, 1, 1)
            else:
                mask_ = mask
        else:
            raise ValueError("Mask must be a list, string (file path), or torch.Tensor.")
        self.mask = mask_
        self.H, self.W = H, W

    def forward(self, x, noise=True):
        B = x.shape[0]
        y = x[self.mask[None]].view(B, -1)
        # add noise
        if noise:
            y = super().forward(y)
        return y

    def pseudo_inv(self, y):
        x = torch.zeros(y.shape[0], 3 * self.H * self.W, dtype=y.dtype, device=y.device)
        x[:, self.mask.view(-1)] = y
        x = x.view(y.shape[0], 3, self.H, self.W)
        # x = inpaint_nearest(x[0], self.mask[0])[None]
        return x


def inpaint_nearest(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Fill missing pixels in an image using the nearest observed pixel value.

    Args:
        image: A tensor of shape [C, H, W] representing the image.
        mask: A tensor of shape [H, W] with 1 for observed pixels and 0 for missing pixels.

    Returns:
        A tensor of shape [C, H, W] where missing pixels have been filled.
    """
    # Move tensors to CPU and convert to numpy arrays.
    image_np = image.cpu().float().numpy()
    # Convert mask to boolean: True for observed, False for missing.
    mask_np = mask.cpu().numpy().astype(bool)

    # Compute the distance transform of the inverse mask (~mask_np).
    # The function returns:
    #   - distances: distance to the nearest True pixel in mask_np
    #   - indices: the indices of that nearest True pixel for each pixel.
    # indices has shape (2, H, W): first row is the row index, second row is the column index.
    _, indices = distance_transform_edt(~mask_np, return_indices=True)

    # Create a copy of the image to hold the filled values.
    filled_image_np = np.empty_like(image_np)

    # For each channel, replace every pixel with the value of the nearest observed pixel.
    for c in range(image_np.shape[0]):
        filled_image_np[c] = image_np[c, indices[0], indices[1]]

    # Convert back to a torch tensor and send to the original device.
    return torch.from_numpy(filled_image_np).to(image.device).to(image.dtype)
    
class MotionBlur(BaseDegradation):
    def __init__(self, kernel_size=5, img_size=256, noise_std=0.0):
        super().__init__(noise_std=noise_std)
        deg_config = munchify({
            'channels': 3,
            'image_size': img_size,
            'deg_scale': kernel_size
        })
        self.img_size = img_size
        self.deg = get_degradation("deblur_motion", deg_config, device="cuda")

    def forward(self, x, noise=True):
        dtype = x.dtype
        y = self.deg.A(x.float())
        # add noise
        if noise:
            y = super().forward(y)
        return y.to(dtype)
    
    def pseudo_inv(self, y):
        dtype = y.dtype
        x = self.deg.At(y.float()).reshape(1,3,self.img_size, self.img_size)
        return x.to(dtype)
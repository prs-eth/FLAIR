import random
from pathlib import Path
import os
import torch
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    learned_perceptual_image_patch_similarity,
)
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import numpy as np
import cv2
from torchvision import transforms


RESAMPLE_MODE = Image.BICUBIC

def load_captions_from_file(caption_file_path, user_prompt, skip=1):
    """Loads captions from a file, handling missing captions.

    Args:
        caption_file_path (str): Path to the caption file.
        user_prompt (str): The base prompt to prepend to captions.

    Returns:
        dict: A dictionary mapping filenames to captions.
    """
    captions = {}
    print(f"Loading captions from file: {caption_file_path}")
    try:
        with open(caption_file_path, "r") as f:
            for i, line in enumerate(f):
                if i % skip != 0:
                    continue
                line = line.strip()
                if not line: # Skip empty lines
                    continue
                parts = line.split(":", 1) # Split by colon only
                if len(parts) == 2:
                    filename, caption = parts
                    caption = caption.strip()
                    # Add user prompt only if caption is not empty, or always add it based on desired logic
                    # Current logic adds user_prompt regardless.
                    captions[filename] = user_prompt + " " + caption
                elif len(parts) == 1:
                    # Handle lines with filename but no caption after colon
                    filename = parts[0].strip()
                    if filename: # Check if filename is not empty
                        captions[filename] = user_prompt # Assign only user prompt
                        print(f"Warning: No caption found for {filename} in {caption_file_path}. Using user prompt only.")
                    else:
                        print(f"Warning: Skipping line with empty filename in {caption_file_path}: {line}")
                else:
                     # This case should ideally not happen with split(":", 1) unless the line is empty (handled above)
                     print(f"Warning: Skipping malformed line in {caption_file_path}: {line}")
    except FileNotFoundError:
        print(f"Error: Caption file not found: {caption_file_path}")
        # Depending on desired behavior, you might want to return an empty dict, 
        # raise an exception, or exit.
        # Returning empty dict for now.
        return {}
    except Exception as e:
        print(f"Error reading caption file {caption_file_path}: {e}")
        return {}
    return captions

def skip_iterator(iterator, skip):
    for i, item in enumerate(iterator):
        if i % skip == 0:
            yield item


def generate_output_structure(output_dir, subfolders=[]):
    """
    Generate a directory structure for the output. and return the paths to the subfolders. as template
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_paths = []
    for subfolder in subfolders:
        output_paths.append(Path(os.path.join(output_dir, subfolder)))
        (output_paths[-1]).mkdir(exist_ok=True, parents=True)
        output_paths[-1] = os.path.join(output_paths[-1], "{}.png")
    return output_paths


def find_files(path, ext="png"):
    if os.path.isdir(path):
        path = Path(path)
        sorted_files = sorted(list(path.glob(f"*.{ext}")))
        return sorted_files
    else:
        return [path]


def load_guidance_image(path, size=768):
    """
    Load an image and convert it to a tensor.
    Args: path to the image
    returns: tensor of the image of shape (1, 3, H, W)
    """
    img = Image.open(path)
    img = img.convert("RGB")
    tf = transforms.Compose([
    transforms.Resize(size),
    transforms.CenterCrop(size),
    transforms.ToTensor()
    ])
    img = tf(img) * 2 - 1
    return img.unsqueeze(0)


def yield_images(path, ext="png", size=None):
    files = find_files(path, ext)
    for file in files:
        yield load_guidance_image(file, size)


def yield_videos(paths, ext="png", H=None, W=None, n_frames=61):
    for path in paths:
        yield read_video(path, H, W, n_frames)


def read_video(path, H=None, W=None, n_frames=61) -> list[Image]:
    path = Path(path)
    frames = []

    if Path(path).is_dir():
        files = sorted(list(path.glob("*.png")))
        for file in files[:n_frames]:
            image = Image.open(file)
            image.load()
            if H is not None and W is not None:
                image = image.resize((W, H), resample=Image.BICUBIC)
            # to tensor
            image = (
                torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)
                / 255.0
                * 2
                - 1
            )
            frames.append(image)
        H, W = frames[0].size()[-2:]
        frames = torch.stack(frames).unsqueeze(0)
        return frames, (10, H, W)

    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        if H is not None and W is not None:
            frame = frame.resize((W, H), resample=Image.BICUBIC)
        # to tensor
        frame = (
            torch.tensor(np.array(frame), dtype=torch.float32).permute(2, 0, 1)
            / 255.0
            * 2
            - 1
        )
        frames.append(frame)

    capture.release()
    # to torch
    frames = torch.stack(frames).unsqueeze(0)
    return frames, (fps, W, H)


def resize_video(
    video: list[Image], width, height, resample_mode=RESAMPLE_MODE
) -> list[Image]:
    frames_lr = []
    for frame in video:
        frame_lr = frame.resize((width, height), resample_mode)
        frames_lr.append(frame_lr)
    return frames_lr


def export_to_video(
    video_frames,
    output_video_path=None,
    fps=8,
    put_numbers=False,
    annotations=None,
    fourcc="mp4v",
):
    fourcc = cv2.VideoWriter_fourcc(*fourcc)  # codec
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, video_frames[0].size)
    for i, frame in enumerate(video_frames):
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        if put_numbers:
            text_position = (frame.shape[1] - 60, 30)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 255, 255)
            line_type = 2
            cv2.putText(
                frame,
                f"{i + 1}",
                text_position,
                font,
                font_scale,
                font_color,
                line_type,
            )

        if annotations:
            annotation = annotations[i]
            frame = draw_bodypose(
                frame, annotation["candidates"], annotation["subsets"]
            )

        writer.write(frame)
    writer.release()


def export_images(frames: list[Image], dir_name):
    dir_name = Path(dir_name)
    dir_name.mkdir(exist_ok=True, parents=True)
    for i, frame in enumerate(frames):
        frame.save(dir_name / f"{i:05d}.png")


def vid2tensor(images: list[Image]) -> torch.Tensor:
    # PIL to numpy
    if not isinstance(images, list):
        raise ValueError()
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)

    if images.ndim == 3:
        # L mode, add luminance channel
        images = np.expand_dims(images, -1)

    # numpy to torch
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images


def compute_metrics(
    source: list[Image],
    output: list[Image],
    output_lq: list[Image],
    target: list[Image],
) -> dict:
    psnr_ab = torch.tensor(
        np.mean(compute_color_metrics(output, target)["psnr_ab"])
    ).float()

    source = vid2tensor(source)
    output = vid2tensor(output)
    output_lq = vid2tensor(output_lq)
    target = vid2tensor(target)

    mse = ((output - target) ** 2).mean()
    psnr = peak_signal_noise_ratio(output, target, data_range=1.0, dim=(1, 2, 3))
    # lpips = learned_perceptual_image_patch_similarity(output, target)

    mse_source = ((output_lq - source) ** 2).mean()
    psnr_source = peak_signal_noise_ratio(
        output_lq, source, data_range=1.0, dim=(1, 2, 3)
    )

    return {
        "mse": mse.detach().cpu().item(),
        "psnr": psnr.detach().cpu().item(),
        "psnr_ab": psnr_ab.detach().cpu().item(),
        "mse_source": mse_source.detach().cpu().item(),
        "psnr_source": psnr_source.detach().cpu().item(),
    }


def compute_psnr_ab(x, y_gt, pp_max=202.33542248):
    """Computes the PSNR of the ab color channels.

    Note that the CIE-Lab space is asymmetric.
    The maximum size for the 2 channels of the ab subspace is approximately 202.3354...
    pp_max: Approximated maximum swing for the ab channels of the CIE-Lab color space
        max_{x \in CIE-Lab} {x_a x_b} - min_{x \in CIE-Lab} {x_a x_b}
    """
    assert (
        len(x.shape) == 3
    ), f"Expecting data of the size HW2 but found {x.shape}; This should be a,b channels of CIE-Lab Space"
    assert (
        len(y_gt.shape) == 3
    ), f"Expecting data of the size HW2 but found {y_gt.shape}; This should be a,b channels of CIE-Lab Space"
    assert (
        x.shape == y_gt.shape
    ), f"Expecting data to have identical shape but found {y_gt.shape} != {x.shape}"

    H, W, C = x.shape
    assert (
        C == 2
    ), f"This function assumes that both x & y are both the ab channels of the CIE-Lab Space"

    MSE = np.sum((x - y_gt) ** 2) / (H * W * C)  # C=2, two channels
    MSE = np.clip(MSE, 1e-12, np.inf)

    PSNR_ab = 10 * np.log10(pp_max**2) - 10 * np.log10(MSE)

    return PSNR_ab


def compute_color_metrics(out: list[Image], target: list[Image]):
    if len(out) != len(target):
        raise ValueError("Videos do not have same length")

    metrics = {"psnr_ab": []}

    for out_frame, target_frame in zip(out, target):
        out_frame, target_frame = np.asarray(out_frame), np.asarray(target_frame)
        out_frame_lab, target_frame_lab = rgb2lab(out_frame), rgb2lab(target_frame)

        psnr_ab = compute_psnr_ab(out_frame_lab[..., 1:3], target_frame_lab[..., 1:3])
        metrics["psnr_ab"].append(psnr_ab.item())

    return metrics


def to_device(sample, device):
    result = {}
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            result[key] = val.to(device)
        elif isinstance(val, list):
            new_val = []
            for e in val:
                if isinstance(e, torch.Tensor):
                    new_val.append(e.to(device))
                else:
                    new_val.append(val)
            result[key] = new_val
        else:
            result[key] = val
    return result


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def adapt_unet(unet, lora_rank=None, in_conv_mode="zeros"):
    # adapt conv_in
    kernel = unet.conv_in.weight.data
    if in_conv_mode == "zeros":
        new_kernel = torch.zeros(320, 4, 3, 3, dtype=kernel.dtype, device=kernel.device)
    elif in_conv_mode == "reflect":
        new_kernel = kernel[:, 4:].clone()
    else:
        raise NotImplementedError
    unet.conv_in.weight.data = torch.cat([kernel, new_kernel], dim=1)
    if in_conv_mode == "reflect":
        unet.conv_in.weight.data *= 2.0 / 3.0
    unet.conv_in.in_channels = 12

    if lora_rank is not None:
        from peft import LoraConfig

        types = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear, torch.nn.Embedding)
        target_modules = [
            (n, m) for n, m in unet.named_modules() if isinstance(m, types)
        ]
        # identify parameters (not modules) that will not be lora'd
        for _, m in target_modules:
            m.requires_grad_(False)
        not_adapted = [p for p in unet.parameters() if p.requires_grad]

        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            # target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            target_modules=[n for n, _ in target_modules],
        )
        # the following line sets all parameters except the loras to non-trainable
        unet.add_adapter(unet_lora_config)

        unet.conv_in.requires_grad_()
        for p in not_adapted:
            p.requires_grad_()


def repeat_infinite(iterable):
    def repeated():
        while True:
            yield from iterable

    return repeated


class CPUAdam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        # keep this in main memory to save VRAM
        self.state = {
            param: {
                "step": 0,
                "exp_avg": torch.zeros_like(param, device="cpu"),
                "exp_avg_sq": torch.zeros_like(param, device="cpu"),
            }
            for param in self.params
        }

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.data.cpu()
            if self.weight_decay != 0:
                grad.add_(param.data, alpha=self.weight_decay)

            state = self.state[param]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            beta1, beta2 = self.betas

            state["step"] += 1

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            denom = exp_avg_sq.sqrt().add_(self.eps)

            step_size = (
                self.lr
                * (1 - beta2 ** state["step"]) ** 0.5
                / (1 - beta1 ** state["step"])
            )

            # param.data.add_((-step_size * (exp_avg / denom)).cuda())

            param.data.addcdiv_(exp_avg.cuda(), denom.cuda(), value=-step_size)

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    def state_dict(self):
        return self.state

    def set_lr(self, lr):
        self.lr = lr

import torchmetrics
import torch
from PIL import Image
import argparse
from flair.utils import data_utils
import os
import tqdm
import torch.nn.functional as F
from torchmetrics.image.kid import KernelInceptionDistance


MAX_BATCH_SIZE = None

@torch.no_grad()
def main(args):
    # Determine device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # load images
    gt_iterator = data_utils.yield_images(os.path.abspath(args.gt), size=args.resolution)
    pred_iterator = data_utils.yield_images(os.path.abspath(args.pred), size=args.resolution)
    fid_metric = torchmetrics.image.fid.FrechetInceptionDistance(normalize=True).to(device)
    # kid_metric = KernelInceptionDistance(subset_size=args.kid_subset_size, normalize=True).to(device)
    lpips_metric = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=False, reduction="mean"
    ).to(device)
    if args.patch_size:
        patch_fid_metric = torchmetrics.image.fid.FrechetInceptionDistance(normalize=True).to(device)
        # patch_kid_metric = KernelInceptionDistance(subset_size=args.kid_subset_size, normalize=True).to(device)
    psnr_list = []
    lpips_list = []
    ssim_list = []
    # iterate over images
    for gt, pred in tqdm.tqdm(zip(gt_iterator, pred_iterator)):
        # Move tensors to the selected device
        gt = gt.to(device)
        pred = pred.to(device)

        # resize gt to pred size
        if gt.shape[-2:] != (args.resolution, args.resolution):
            gt = F.interpolate(gt, size=args.resolution, mode="area")
        if pred.shape[-2:] != (args.resolution, args.resolution):
            pred = F.interpolate(pred, size=args.resolution, mode="area")
        # to range [0,1]
        gt_norm = gt * 0.5 + 0.5
        pred_norm = pred * 0.5 + 0.5
        # compute PSNR
        psnr = torchmetrics.functional.image.peak_signal_noise_ratio(
            pred_norm, gt_norm, data_range=1.0
        )
        psnr_list.append(psnr.cpu()) # Move result to CPU
        # compute LPIPS
        lpips_score = lpips_metric(pred.clip(-1,1), gt.clip(-1,1))
        lpips_list.append(lpips_score.cpu()) # Move result to CPU
        # compute SSIM
        ssim = torchmetrics.functional.image.structural_similarity_index_measure(
            pred_norm, gt_norm, data_range=1.0
        )
        ssim_list.append(ssim.cpu()) # Move result to CPU
        print(f"PSNR: {psnr}, LPIPS: {lpips_score}, SSIM: {ssim}")
        # compute FID
        # Ensure inputs are on the correct device (already handled by moving gt/pred earlier)
        fid_metric.update(gt_norm, real=False)
        fid_metric.update(pred_norm, real=True)
        # compute KID
        # kid_metric.update(pred, real=False)
        # kid_metric.update(gt, real=True)
        # compute Patchwise FID/KID if patch_size is specified
        if args.patch_size:
            # Extract patches
            patch_size = args.patch_size
            gt_patches = F.unfold(gt_norm, kernel_size=patch_size, stride=patch_size)
            pred_patches = F.unfold(pred_norm, kernel_size=patch_size, stride=patch_size)
            # Reshape patches: (B, C*P*P, N_patches) -> (B*N_patches, C, P, P)
            B, C, H, W = gt.shape
            N_patches = gt_patches.shape[-1]
            gt_patches = gt_patches.permute(0, 2, 1).reshape(B * N_patches, C, patch_size, patch_size)
            pred_patches = pred_patches.permute(0, 2, 1).reshape(B * N_patches, C, patch_size, patch_size)
            # Update patch FID metric (inputs are already on the correct device)
            # Update patch KID metric
            # process mini batches of patches
            if MAX_BATCH_SIZE is None:
                patch_fid_metric.update(pred_patches, real=False)
                patch_fid_metric.update(gt_patches, real=True)
                # patch_kid_metric.update(pred_patches, real=False)
                # patch_kid_metric.update(gt_patches, real=True)
            else:
                for i in range(0, N_patches, MAX_BATCH_SIZE):
                    patch_fid_metric.update(pred_patches[i:i + MAX_BATCH_SIZE], real=False)
                    patch_fid_metric.update(gt_patches[i:i + MAX_BATCH_SIZE], real=True)
                    # patch_kid_metric.update(pred_patches[i:i + MAX_BATCH_SIZE], real=False)
                    # patch_kid_metric.update(gt_patches[i:i + MAX_BATCH_SIZE], real=True)

    # compute FID
    fid = fid_metric.compute()
    # compute KID
    # kid_mean, kid_std = kid_metric.compute()
    if args.patch_size:
        patch_fid = patch_fid_metric.compute()
        # patch_kid_mean, patch_kid_std = patch_kid_metric.compute()
    # compute average metrics (on CPU)
    avg_psnr = torch.mean(torch.stack(psnr_list))
    avg_lpips = torch.mean(torch.stack(lpips_list))
    avg_ssim = torch.mean(torch.stack(ssim_list))
    # compute standard deviation (on CPU)
    std_psnr = torch.std(torch.stack(psnr_list))
    std_lpips = torch.std(torch.stack(lpips_list))
    std_ssim = torch.std(torch.stack(ssim_list))
    print(f"PSNR: {avg_psnr} +/- {std_psnr}")
    print(f"LPIPS: {avg_lpips} +/- {std_lpips}")
    print(f"SSIM: {avg_ssim} +/- {std_ssim}")
    print(f"FID: {fid}") # FID is computed on the selected device, print directly
    # print(f"KID: {kid_mean} +/- {kid_std}") # KID is computed on the selected device, print directly
    if args.patch_size:
        print(f"Patch FID ({args.patch_size}x{args.patch_size}): {patch_fid}") # Patch FID is computed on the selected device, print directly
        # print(f"Patch KID ({args.patch_size}x{args.patch_size}): {patch_kid_mean} +/- {patch_kid_std}") # Patch KID is computed on the selected device, print directly
    # save to prediction folder
    out_file = os.path.join(args.pred, "fid_metrics.txt")
    with open(out_file, "w") as f:
        f.write(f"PSNR: {avg_psnr.item()} +/- {std_psnr.item()}\n") # Use .item() for scalar tensors
        f.write(f"LPIPS: {avg_lpips.item()} +/- {std_lpips.item()}\n")
        f.write(f"SSIM: {avg_ssim.item()} +/- {std_ssim.item()}\n")
        f.write(f"FID: {fid.item()}\n") # Use .item() for scalar tensors
        # f.write(f"KID: {kid_mean.item()} +/- {kid_std.item()}\n") # Use .item() for scalar tensors
        if args.patch_size:
            f.write(f"Patch FID ({args.patch_size}x{args.patch_size}): {patch_fid.item()}\n") # Use .item() for scalar tensors
            # f.write(f"Patch KID ({args.patch_size}x{args.patch_size}): {patch_kid_mean.item()} +/- {patch_kid_std.item()}\n") # Use .item() for scalar tensors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics")
    parser.add_argument("--gt", type=str, help="Path to ground truth image")
    parser.add_argument("--pred", type=str, help="Path to predicted image")
    parser.add_argument("--resolution", type=int, default=768, help="resolution at which to evaluate")
    parser.add_argument("--patch_size", type=int, default=None, help="Patch size for Patchwise FID/KID computation (e.g., 12). If None, skip.")
    parser.add_argument("--kid_subset_size", type=int, default=1000, help="Subset size for KID computation.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run computation on (cpu or cuda)")
    args = parser.parse_args()

    main(args)

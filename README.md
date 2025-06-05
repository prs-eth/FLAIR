<div align="center">

# FLAIR: Flow-Based Latent Alignment for Image Restoration

**Julius Erbach<sup>1</sup>, Dominik Narnhofer<sup>1</sup>, Andreas Dombos<sup>1</sup>, Jan Eric Lenssen<sup>1</sup>, Bernt Schiele<sup>2</sup>, Konrad Schindler<sup>1</sup>**  
<br>
<sup>1</sup> Photogrammetry and Remote Sensing, ETH Zurich  
<sup>2</sup> Max Planck Institute for Informatics, Saarbrücken  

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2506.02680)
[![Page](https://img.shields.io/badge/Project-Page-green)](https://inverseFLAIR.github.io)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/prs-eth/FLAIR)
</div>

<p align="center">
  <img src="assets/teaser3.svg" alt="teaser" width=98%"/>
</p>
<p align="center">
  <emph>FLAIR</emph> is a novel approach for solving inverse imaging problems using flow-based posterior sampling.
</p>

## Installation

1.  Clone the repository:
    ```bash
    git clone git@github.com:prs-eth/FLAIR.git
    cd FLAIR
    ```

2.  Create a virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  
    ```

3.  Install the required dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    pip install .
    ```

## Running Inference

To run inference, you can use one of the Python script run_image_inv.py with the according config file.
An example from the FFQH dataset
```bash
python inference_scripts/run_image_inv.py --config configs/inpainting.yaml --target_file examples/girl.png --result_folder output --prompt="a high quality photo of a face"
```
Or an example from the DIV2K dataset with captions provided by DAPE using the degraded input. The masks can be defined as rectanlge coordinates in the config file or provided as .npy file where true pixels are observed and false are masked out.

```bash
python inference_scripts/run_image_inv.py --config configs/inpainting.yaml --target_file examples/sunflowers.png --result_folder output --prompt="a high quality photo of bloom, blue, field, flower, sky, sunflower, sunflower field, yellow" --mask_file DIV2k_mask.npy
```

```bash
python inference_scripts/run_image_inv.py --config configs/x12.yaml --target_file examples/sunflowers.png --result_folder output --prompt="a high quality photo of bloom, blue, field, flower, sky, sunflower, sunflower field, yellow"
```

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{er2025solving,
  title={Solving Inverse Problems with FLAIR},
  author={Erbach, Julius and Narnhofer, Dominik and Dombos, Andreas and Lenssen, Jan Eric and Schiele, Bernt and Schindler, Konrad},
  journal={arXiv},
  year={2025}
}
```

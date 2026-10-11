# Dead Tree Segmentation from Aerial RGB + NIR Imagery

UNSW COMP9517 (Computer Vision), Term 2 2025 — group project (5 members), report dated 8 August 2025.

**中文简介**：航拍图像中枯死树（standing dead tree）的语义分割。小组报告比较了传统机器学习（KNN/决策树/随机森林）和深度模型（U-Net、ResNet34-U-Net、PSPNet、改进的 DeepLabV3+）。本仓库只包含我这一部分里**能提供代码证据**的内容：RGB + 近红外（NIR）四通道 DeepLabV3+（ResNet34）基线的 Colab 笔记本。小组报告中的 NIR 注意力变体的训练代码不在本仓库（见下方"已知缺口"）。

## My contribution

I worked on the DeepLabV3+ branch of the group project: the notebook in this repository (4-channel RGB+NIR input, ResNet34 encoder, focal + Dice loss, validation-IoU early stopping). The other branches in the group report — classical ML baselines, U-Net / ResNet34-U-Net and PSPNet — were done by other members and are not part of this repository.

## What is here

`deeplabv3plus_rgb_nir_baseline.ipynb` – training and evaluation of a binary dead-tree segmentation model, with stored outputs:

- **Data**: the Kaggle dataset "Aerial Imagery for Standing Dead Tree Segmentation" (RGB, NRG and mask PNGs; not included).
- **Input**: RGB with ImageNet statistics plus one NIR channel taken from the NRG image, stacked into 4 channels, resized to 384×384; augmentation with flips.
- **Model**: DeepLabV3+ (`segmentation_models_pytorch`), ResNet34 encoder with ImageNet weights; the first convolution is replaced by a 4-channel version (22.44 M parameters).
- **Loss / optimiser**: focal loss (α = 0.25, γ = 2) + Dice loss; AdamW (lr 1e-4, weight decay 1e-4), cosine annealing to 1e-6, batch size 4 (2 on CPU), up to 25 epochs, early stopping (patience 5) on validation IoU.
- **Result printed by the notebook**: best validation IoU 0.2370; **test IoU 0.2404, Dice 0.3832**.
- A final cell plots typical spectral signatures of dead wood and other materials to illustrate why the NIR band helps.

## Important caveats (read before quoting the numbers)

1. **Only 127 images were used.** In the Drive copy the notebook ran on, just 127 of the 444 images had all three files (RGB, NRG, mask); the other 317 were skipped by the notebook's completeness check. The split is therefore 101 train / 13 validation / 13 test (`random_state = 42`). With a 13-image test set the 0.2404 test IoU is a very noisy estimate (validation IoU swings between about 0.01 and 0.24 from epoch to epoch).
2. **These numbers are not the group report's numbers.** The report evaluates on the full 444-image dataset with a random 80/20 train/validation split.

## Known gaps: the group report's attention ablation

The group report (Enhanced DeepLabV3+ section) describes an NIR attention module and a Focal + Dice + Boundary loss, and gives this ablation on validation IoU: RGB-only 0.2145 → + NIR channel 0.2389 → + NIR attention 0.2762 (+28.8% relative). **The training code that produced the attention variant is not in this repository** — the notebook here corresponds to the "+ NIR channel" step (with focal + Dice loss), and its validation IoU (0.2370) is close to that step's 0.2389. I am including this explicitly so the repository does not overstate what it can verify. <!-- TODO(owner): add the notebook/script for the NIR-attention variant if it can be found (other Colab account / machine / teammate). -->

## How to reproduce

1. Download the Kaggle dataset and arrange it as `USA_segmentation/{RGB_images,NRG_images,masks}` with files `RGB_<id>.png`, `NRG_<id>.png`, `mask_<id>.png`. Upload it to Google Drive at `MyDrive/archive/USA_segmentation` (or edit `BASE_PATH` in the notebook).
2. Open the notebook in Google Colab with a GPU runtime (the stored outputs were produced on CUDA) and mount Drive.
3. `pip install segmentation-models-pytorch albumentations opencv-python scikit-learn tqdm` if Colab does not already provide them, then Run all. The notebook saves the best weights as `best_robust_model.pth`.
4. If all 444 images are present, the split will be 355 / 44 / 45 instead of 101 / 13 / 13 and the numbers will differ from the stored output; that is expected and is the more meaningful experiment.

## Likely interview questions

**Why add NIR as a fourth channel?** Dead trees lose chlorophyll, which changes their near-infrared reflectance relative to healthy vegetation; NIR gives a cue that RGB lacks. The final cell of the notebook plots the spectral signatures that motivate this.

**How do you feed 4 channels into an ImageNet-pretrained ResNet?** The first convolution is replaced by a 4-channel convolution: the pretrained RGB filters are copied to the first three input channels and the NIR channel's filter is initialised as the mean of the three RGB filters (model class in the notebook).

**Why focal + Dice loss?** Dead-tree pixels are a small fraction of each image; focal loss down-weights easy background pixels and Dice directly optimises overlap.

**Why are the IoU values so low?** The task is hard (small, sparse targets; 444 images, mostly from a single region), and in this notebook the training set is only 101 images. The group report reached 0.2762 validation IoU with the attention variant on the full set — still a modest score, and I would describe it as a proof of concept rather than a deployable model.

**What would you do next?** Use the complete dataset, k-fold cross-validation instead of a 13-image test set, report per-image IoU with confidence intervals, and ablate NIR attention and the boundary loss separately.

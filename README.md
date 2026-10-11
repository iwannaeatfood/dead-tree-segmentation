# Dead Tree Segmentation from Aerial RGB + NIR Imagery

UNSW COMP9517 (Computer Vision), Term 2 2025 — group project (5 members), report dated 8 August 2025.

**中文简介**：航拍图像中枯死树（standing dead tree）的语义分割。小组报告比较了传统机器学习（KNN/决策树/随机森林）和深度模型（U-Net、ResNet34-U-Net、PSPNet、改进的 DeepLabV3+）。本仓库包含我负责的 DeepLabV3+ 分支中的 RGB + 近红外（NIR）四通道基线 Colab 笔记本；小组报告中 NIR 注意力变体的训练代码不在本仓库。

## My part

The DeepLabV3+ branch of the group project: the notebook in this repository (4-channel RGB+NIR input, ResNet34 encoder, focal + Dice loss, validation-IoU early stopping). The other branches in the group report — classical ML baselines, U-Net / ResNet34-U-Net and PSPNet — were done by other members and are not part of this repository.

## What is here

`deeplabv3plus_rgb_nir_baseline.ipynb` – training and evaluation of a binary dead-tree segmentation model, with stored outputs:

- **Data:** Kaggle "Aerial Imagery for Standing Dead Tree Segmentation" (RGB, NRG and mask PNGs; not included).
- **Input:** RGB with ImageNet statistics plus one NIR channel from the NRG image, stacked into 4 channels, resized to 384×384; flip augmentation.
- **Model:** DeepLabV3+ (`segmentation_models_pytorch`), ResNet34 encoder with ImageNet weights, 4-channel first convolution (RGB filters copied, NIR filter initialised as the mean of the three), 22.44 M parameters.
- **Training:** focal (α = 0.25, γ = 2) + Dice loss; AdamW (lr 1e-4, weight decay 1e-4), cosine annealing, batch size 4, up to 25 epochs, early stopping (patience 5) on validation IoU.
- **Result printed by the notebook:** best validation IoU 0.2370; test IoU 0.2404, Dice 0.3832.
- A final cell plots typical spectral signatures of dead wood and other materials.

## Notes on the numbers

- **Subset used:** in the Drive copy the notebook ran on, 127 of the 444 images had all three files (RGB, NRG, mask); the others were skipped by the notebook's completeness check. The split is 101 train / 13 validation / 13 test (`random_state = 42`), so the test IoU comes from 13 images and validation IoU varies between about 0.01 and 0.24 from epoch to epoch.
- **Group report:** the report evaluates on the full 444-image dataset with an 80/20 train/validation split and gives an ablation on validation IoU of RGB-only 0.2145 → + NIR channel 0.2389 → + NIR attention 0.2762. The notebook here corresponds to the "+ NIR channel" step (validation IoU 0.2370); the training code for the NIR-attention variant is not in this repository.

## How to reproduce

1. Download the Kaggle dataset and arrange it as `USA_segmentation/{RGB_images,NRG_images,masks}` with files `RGB_<id>.png`, `NRG_<id>.png`, `mask_<id>.png`; upload it to Google Drive at `MyDrive/archive/USA_segmentation` (or edit `BASE_PATH` in the notebook).
2. Open the notebook in Google Colab with a GPU runtime and mount Drive.
3. Install `segmentation-models-pytorch albumentations opencv-python scikit-learn tqdm` if Colab does not provide them, then run all. The best weights are saved as `best_robust_model.pth`.

With all 444 images present the split becomes 355 / 44 / 45 and the numbers will differ from the stored output.

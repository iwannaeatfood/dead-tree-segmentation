# Dead Tree Segmentation from Aerial RGB + NIR Imagery

UNSW COMP9517 (Computer Vision), Term 2 2025 — group project (5 members).

**中文简介**：航拍图像枯死树语义分割。本仓库包含 RGB + 近红外（NIR）四通道 DeepLabV3+ 基线实验的 Colab 笔记本；小组报告中的其他模型与对比实验不在此仓库。

## What is here

`deeplabv3plus_rgb_nir_baseline.ipynb` – training and evaluation of a binary dead-tree segmentation model:

- **Data**: aerial imagery with an NIR band, binary dead-tree masks (the notebook reads them from `archive/USA_segmentation` on Google Drive; the data is not included).
- **Input**: RGB with ImageNet statistics plus one NIR channel (simple standardisation), stacked into 4 channels.
- **Model**: DeepLabV3+ (segmentation_models_pytorch), ResNet34 encoder with ImageNet weights; the first convolution is replaced by a 4-channel version.
- **Training**: AdamW (weight decay 1e-4), cosine annealing, batch size 4, up to 25 epochs with early stopping on validation IoU; IoU/Dice tracked per epoch.
- **Result (as printed by the notebook)**: best validation IoU 0.2370; **test IoU 0.2404, test Dice 0.3832**. Validation IoU is noisy between epochs (0.01–0.24).
- A final cell plots typical spectral signatures of dead wood and other materials to explain why the NIR band helps.

The group's other models (e.g. U-Net, classical ML baselines) and the full report are not part of this repository.

# Pixel-wise Weighting Network for Few-Shot PCB Defect Detection

Official implementation of the paper **"Pixel-wise weighting network for few-shot PCB
defect detection in personalized small-batch production"** (submitted to *Scientific Reports*).

## Repository structure

|File|Content|
|-|-|
|`preprocess\_diff.py`|Geometric registration (Sec. 2.2 of the paper) + difference-map generation, threshold masking (T from the p-th percentile of defect-free samples), and cross-differencing data augmentation (Sec. 2.3).|
|`model.py`|DWCN network (position-specific pixel-wise weighting layer) and the loss function of Eqs. (12)-(13).|
|`train.py`|Training script: Adam (lr = 1e-3) + cosine annealing, 100 epochs, equal positive/negative batches, model selection on the validation set every 10 epochs.|
|`evaluate.py`|Evaluation script: frozen-threshold protocol (min-max bounds and decision threshold from the validation set only, exhaustive search over candidate cut points), reporting the confusion matrix and Accuracy / Precision / Recall / F1.|

The implementation details of all baseline methods compared in the paper are documented in
Supplementary Table S1-1 of the manuscript.

## Requirements

* Python >= 3.8
* PyTorch, torchvision
* OpenCV (`opencv-python`), NumPy, Pillow, tqdm

## Data

Input images are RGB board images; output difference maps follow the naming convention
`<name>\_0.jpg` (defective) / `<name>\_1.jpg` (non-defective), which `train.py` /
`evaluate.py` read directly.

## Usage

```bash
# 1) Registration + difference-map generation (Sec. 2.2-2.3)
#    --std: defect-free standard reference board; --good\_dir/--bad\_dir: board image folders;
#    --partners: cross-differencing partners per image (3-5); --percentile: percentile for T;
#    --roi: optional fixed ROI on the reference image (x,y,w,h), 350x280 in the paper.
python preprocess\_diff.py --std y1.jpg --good\_dir good\_boards --bad\_dir bad\_boards \\
    --out\_dir diff\_dataset --partners 4 --percentile 97 --roi 0,0,350,280

# 2) Training (60 training images: 30+30; 40 validation images: 20+20)
python train.py --train\_dir train60 --val\_dir val40 --out dwcn\_best.pth

# 3) Evaluation on the cross-batch test set (400 images)
python evaluate.py --model dwcn\_best.pth --val\_dir val40 --test\_dir test400
```


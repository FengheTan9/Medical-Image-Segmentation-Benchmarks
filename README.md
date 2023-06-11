# Medical 2D Image Segmentation Benckmark





For easy evaluation and fair comparison on 2d medical image segmentation method, we aim to collect and build a medical image segmentation U-shape architecture benchmark to implement the medical 2d image segmentation tasks.

This repo has collected and re-implemented medical image segmentation network based on U-shape architecture are followed:

| Network         | Original code                                                | Reference                                                    |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| U-Net           | [Caffe](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net) | [MICCAI](https://arxiv.org/pdf/1505.04597.pdf)               |
| Attention U-Net | [Pytorch](https://github.com/ozan-oktay/Attention-Gated-Networks) | [Arxiv](https://arxiv.org/pdf/1804.03999.pdf)                |
| UNet++          | [Pytorch](https://github.com/MrGiovanni/UNetPlusPlus)        | [MICCAI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7329239/pdf/nihms-1600717.pdf) |
| UNet3+          | [Pytorch](https://github.com/ZJUGiveLab/UNet-Version)        | [ICASSP](https://arxiv.org/pdf/2004.08790)                   |
| UNeXt           | [Pytorch](https://github.com/jeya-maria-jose/UNeXt-pytorch)  | [MICCAI](https://arxiv.org/pdf/2203.04967.pdf)               |
| CMUNet          | [Pytorch](https://github.com/FengheTan9/CMU-Net)             | [ISBI](https://arxiv.org/abs/2210.13012)                     |
| TransUnet       | [Pytorch](https://github.com/Beckschen/TransUNet)            | [Arxiv](https://arxiv.org/pdf/2102.04306.pdf)                |
| MedT            | [Pytorch](https://github.com/jeya-maria-jose/Medical-Transformer) | [MICCAI](https://arxiv.org/pdf/2102.10662.pdf)               |
| SwinUnet        | [Pytorch](https://github.com/HuCaoFighting/Swin-Unet)        | [ECCV](https://arxiv.org/pdf/2105.05537.pdf)                 |

## Datasets

Please put the [BUSI](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset) dataset or your own dataset as the following architecture. 

```
├── CMUNet
    ├── inputs
        ├── BUSI
            ├── images
            |   ├── benign (10).png
            │   ├── malignant (17).png
            │   ├── normal (14).png
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── benign (10).png
                |   ├── malignant (17).png
                |   ├── normal (14).png
                |   ├── ...
        ├── your 2D dataset
            ├── images
            |   ├── 0a7e06.png
            │   ├── 0aab0a.png
            │   ├── 0b1761.png
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
```

## Environments

- GPU: NVIDIA GeForce RTX4090 GPU
- Pytorch: 1.13.0 cuda 11.7
- cudatoolkit: 11.7.1
- scikit-learn: 1.0.2

## Training

You can first spilt your dataset:

```python
python spilt.py
```

Then, training and validating your dataset:

```python
python main.py --model CMUNet --base_lr 0.01 --epoch 300 --batch_size 4 --img_size 256 --num_classes 1
```

## Result on BUSI

We train the U-shape networks with [BUSI dataset](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset). The BUSI collected 780 breast ultrasound images, including normal, benign and malignant cases of breast cancer with their corresponding segmentation results. **We only used benign and malignant images (647 images)**. And we randomly split thrice, 70% for training and 30% for validation. In addition, we resize all the images 256×256 and perform random rotation and flip for data augmentation.

|     Method      | Params (M) |    FPS     |  GFLOPs  |      IoU       |    F1-value    |
| :-------------: | :--------: | :--------: | :------: | :------------: | :------------: |
|      U-Net      |   34.52    |   139.32   |  65.52   |   68.61±2.86   |   76.97±3.10   |
| Attention U-Net |   34.87    |   129.92   |  66.63   |   68.55±3.22   |   76.88±3.50   |
|     U-Net++     |   26.90    |   125.50   |  37.62   |   69.49±2.94   |   78.06±3.25   |
|     U-Net3+     |   26.97    |   50.60    |  199.74  |   68.38±3.35   |   76.88±3.68   |
|     CMU-Net     |   49.93    |   93.19    |  91.25   | **71.42±2.65** |   79.49±2.92   |
|    TransUnet    |   105.32   |   112.95   |  38.52   |   71.39±2.37   | **79.85±2.59** |
|      MedT       |  **1.37**  |   22.97    |   2.40   |   63.36±1.56   |   73.37±1.63   |
|    SwinUnet     |   27.14    |   392.21   |   5.91   |   54.11±2.29   |   65.46±1.91   |
|      UNeXt      |    1.47    | **650.48** | **0.58** |   65.04±2.71   |   74.16±2.84   |

## Acknowledgements:

This code-base uses helper functions from [CMU-Net](https://github.com/FengheTan9/CMU-Net)and [Image_Segmntation]([LeeJunHyun/Image_Segmentation: Pytorch implementation of U-Net, R2U-Net, Attention U-Net, and Attention R2U-Net. (github.com)](https://github.com/LeeJunHyun/Image_Segmentation)).

## Other QS:

If you have any questions or suggestions about this project, please contact me through email: 543759045@qq.com
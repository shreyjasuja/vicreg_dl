# Deep Learning Final Project

## Overview:
This project focuses on the implementation and evaluation of
Variance-Invariance-Covariance Regularization (VICReg), a
Self-Supervised Learning (SSL) method. The methodology
employs a joint embedding architecture, a loss function
incorporating invariance, variance, and covariance terms,
and a Siamese-based VICReg network for training.

## Objective 1: SSL training using VICReg on CIFAR Dataset
We used the custom resnet model architecture that we created in the [Mini-Project](https://github.com/shreyjasuja/cifar_10_custom_resnet).

After training the model on CIFAR100 using self-supervised learning, we evaluated the model on 10% CIFAR10 dataset.

These accuracies might seem lower than standard at first glance but we need to remember that these models were fine-tuned only on 1% and 10% of the low-resolution CIFAR10 dataset. 
For comparison, refer to the untrained backbone accuracies which prove that the SSL indeed gave the network more capability for the downstream task.

Step 1: Download [exp folder](https://drive.google.com/drive/folders/1YvShKyGbXxjqYAgbzIsw_lyoyLXMsKYM?usp=share_link) and move it inside objective-1-CIFAR-SSL-training folder.

Step 2: Install dependencies via 
```
pip install torch torchvision matplotlib seaborn scikit-learn pandas
```
Step 3: For Training: 
```
python main_vicreg.py --batch-size 512 --mlp '512-512-512'
```
Step 4: Run the notebook ssl_eval.ipynb to reproduce the results.

[Add accuracies table screenshot]

![download](https://github.com/shreyjasuja/vicreg_dl/assets/30201131/0228baeb-859c-49b8-9e4c-74f06e30507c)

## Objective 2: Fine tuning trained SSL model on LFW dataset
We utilized the official pre-trained ResNet 50 network that
was originally trained on Imagenet, and fine-tuned it on the
”Labeled Faces in the Wild” (LFW) dataset. The objective is
to apply the learned SSL representations to the face recogni-
tion task.

Step 1: Download [exp folder](https://drive.google.com/drive/folders/1LB_KcYa3bsCKaBAOKTaEn1WbCVj9PmXx?usp=share_link) and move it inside objective-1-CIFAR-SSL-training folder.

Step 2: Install dependencies via 
```
pip install torch torchvision matplotlib seaborn scikit-learn pandas
```

Step 3: Run the notebooks lfw_ssl.ipynb and lfw_eval.ipynb to reproduce the results.

Result of grouping similar faces using ANNOY:

![WhatsApp Image 2023-05-14 at 6 02 18 PM](https://github.com/shreyjasuja/vicreg_dl/assets/30201131/feeeb9af-3d76-47f0-a953-d123a7b72a14)

Result of grouping dissimilar faces using ANNOY:

![WhatsApp Image 2023-05-14 at 6 02 30 PM](https://github.com/shreyjasuja/vicreg_dl/assets/30201131/e233ea5f-4cda-4dd8-98f5-ec41e0b36e70)

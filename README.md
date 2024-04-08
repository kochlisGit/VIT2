# ViT2 - Pre-training Vision Transformers for Visual Times Series Forecasting

Computer Vision has witnessed remarkable advancements through the utilization of large Transformer architectures, such as Vision Transformer (ViT). These models achieve impressive performance and generalization capability when trained on large datasets and can be fine-tuned on custom image datasets through transfer learning techniques. On the other hand, time series forecasting models have struggled to achieve a similar level of generalization across diverse datasets. This paper presents ViT$2, a framework composed of four modules, that addresses probabilistic price forecasting and generalization for cryptocurrency markets. The first module injects Gaussian noise into time series data to increase samples availability. The second module transforms the time series data into visual data, using Gramian Angular Fields. The third module converts the ViT architecture into a probabilistic forecasting model. Finally, the fourth module employs Transfer Learning and fine-tuning techniques to enhance its performance on low-resource datasets. Our findings reveal that ViT2 outperforms State-Of-The-Art time series forecasting models across the majority of the datasets evaluated, highlighting the potential of Computer Vision models in the probabilistic time series forecasting domain. The code and models are publicly available at: \url{ https://github.com/kochlisGit/VIT2}.

# Architecture

ViT$2, which is composed in four modules. The first module applies data augmentation via Gaussian noise injection into candlesticks data, further increasing data availability. The second module applies the GAF method to transform every time series features into images, which are then stacked together along the depth axis to form multi-channel images. The third module modifies the original ViT architecture to be compatible with the multi-channel input images and converts it into a probabilistic forecasting model. Then, it trains the modified architecture using the constructed images. The final module applies Transfer Learning and fine-tunes the trained model on a low-resource datasets. TIMM library has been used to implement ViT model architecture (https://timm.fast.ai/).

![https://github.com/kochlisGit/VIT2/blob/main/figs/methodology.jpeg](https://github.com/kochlisGit/VIT2/blob/main/figs/methodology.jpeg)

# Probabilistic Forecasting VIT

![https://github.com/kochlisGit/VIT2/blob/main/figs/vit-modified.jpeg](https://github.com/kochlisGit/VIT2/blob/main/figs/vit-modified.jpeg)

# Comparison

![https://github.com/kochlisGit/VIT2/blob/main/figs/table5.png](https://github.com/kochlisGit/VIT2/blob/main/figs/table5.png)

# Requirements

* python >= 3.10
* matplotlib
* scikit-learn
* numpy==1.26.4
* pandas==2.2.1
* pytorch_lightning==2.1.2
* pyts==0.13.0
* scikit_learn==1.3.2
* ta==0.11.0
* torch==2.1.1
* torchview==0.2.6
* tqdm==4.66.1
* darts
* timm
* pyts

# Probabilistic Time Series Forecasting

This repository uses `DARTS` library to utilize probablistic time-series forecasting models (https://unit8co.github.io/darts/). 

The models used in this repo are:

1. Temporal Fusion Transformer (TFT)
2. N-BEATS
3. NHits

Refer to DARTS library for more information.

# Train Models

Train each model individually by running `python name_train.py`, where name corresponds to the selected model (ForCNN-Resnet, GAF-ViT, TFT, N-BEATS, NHits). Then, after you train all models, you can generate the prediction plots. Notebooks with examples have also been provided.

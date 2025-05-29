# Breast Imaging Preprocessing & UNet Training Pipelines

This repository contains the code accompanying the paper:  
**"[Title of Your Paper]"**  
Submitted to [Conference/Journal Name], 2025.

We focus on evaluating the impact of different preprocessing strategies for breast imaging using two datasets:  
- **CBIS-DDSM (2D mammography)** (https://www.cancerimagingarchive.net/collection/cbis-ddsm/)
- **Duke Breast Cancer MRI (3D MRI)** (https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/)

## 🧠 Key Features

- Training pipelines using **MONAI** and **PyTorch**
- 5-fold cross-validation for robustness
- Easily configurable preprocessing pipelines via JSON
- Automatic GPU scheduling via `cuda.json`
- Support for image–segmentation pair CSVs
- **Dockerfile included for full reproducibility**

## 📁 Dataset Preparation

### 1. CBIS-DDSM
Download from The Cancer Imaging Archive (TCIA):  
https://www.cancerimagingarchive.net/

### 2. Duke Breast MRI
Also available on TCIA:  
https://www.cancerimagingarchive.net/

Prepare a CSV file with the format:

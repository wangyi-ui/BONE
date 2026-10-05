# BONE: Bridging Optimization and Neural Networks for Efficient Multi-View Clustering

This repository contains the official PyTorch implementation of **BONE**, a lightweight and interpretable multi-view clustering (MVC) framework that **Bridges Optimization and Neural networks for Efficient MVC**, as described in our AAAI 2026 paper:

> **BONE: Bridging Optimization and Neural Networks for Efficient Multi-View Clustering**  
> (AAAI 2026)

BONE leverages a **classical optimization backbone** to extract low-level multi-view features and then uses a **small learnable network** to refine these features. This design achieves clustering performance comparable to or even better than existing deep MVC methods, while using **only about 1% of their parameters**, without any pre-training or post-processing.

## Requirements

- numpy==1.24.3
- scikit_learn==1.3.0
- scipy==1.11.1
- torch==2.9.1

## Running Environment in Paper

- **Operating System:** Windows 11
- **CPU:** R7 7840HS
- **RAM:** 64G
- **GPU:** NVIDIA GeForce RTX 4050

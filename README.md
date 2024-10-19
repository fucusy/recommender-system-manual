# PyTorch Recommender System

This project implements a Recommender System using PyTorch. The system is designed to provide recommendations based on user-item interactions, with various algorithms, including collaborative filtering and matrix factorization models.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
    - [Install Conda](#install-conda)
    - [Setup Conda Environment](#setup-conda-environment)
3. [Installation](#installation)
4. [Downloading Recommender System Datasets](#downloading-datasets)
    - [MovieLens Dataset](#movielens-dataset)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Contributing](#contributing)

## Project Overview

This project uses deep learning techniques to implement a recommender system. It supports training on classical datasets such as MovieLens. The PyTorch library is used for the underlying model training and evaluation.

## Environment Setup

### Install Conda

If you don't already have Conda installed, you can download it from the [official Conda website](https://docs.conda.io/en/latest/miniconda.html) and follow the installation instructions for your operating system.

### Setup Conda Environment

1. Create a new Conda environment for this project (you can name it `recommender_env`):

    ```bash
    conda create -n recommender_env python=3.9
    ```

2. Activate the environment:

    ```bash
    conda activate recommender_env
    ```

## Installation

After setting up the Conda environment, follow the steps below to install the necessary libraries:

1. First, install PyTorch. You can install the appropriate version of PyTorch based on your system and CUDA availability. For example for Mac OS:

    - For CPU-only support:

      ```bash
      pip install torch torchvision torchaudio
      ```

2. Install additional required Python libraries:

    ```bash
    pip install numpy pandas scikit-learn matplotlib
    ```

3. Optionally, install Jupyter for notebook support:

    ```bash
    pip install jupyter
    ```

## Downloading Datasets

To train and evaluate the recommender system, we need datasets like MovieLens. Follow the steps below to download the MovieLens dataset:

### MovieLens Dataset

MovieLens is a classical dataset used for recommender system research. You can download it by following these steps:

1. Visit the [MovieLens dataset website](https://grouplens.org/datasets/movielens/).
2. Select the version of the dataset you'd like to download. For example, the **MovieLens 100K** dataset is a good starting point.
3. Download the dataset and unzip the file in your project directory. You can also use the following commands in the terminal to download it directly:

   ```bash
   curl -O http://files.grouplens.org/datasets/movielens/ml-100k.zip
   unzip ml-100k.zip -d data/


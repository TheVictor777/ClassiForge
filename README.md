# ClassiForge 🖼️

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A flexible and powerful PyTorch framework for rapid development of image classification models. ClassiForge is designed to be a reusable and configurable pipeline, handling everything from data loading and augmentation to multi-GPU training and model saving.

This framework allows you to go from a structured image dataset to a trained, high-performance model with just a few lines of configuration.

---

## 📋 Core Features

* **Model Agnostic:** Easily swap between different pre-trained architectures like `ResNet` and `Vision Transformers (ViT)`.
* **Config-Driven:** All training parameters (epochs, learning rate, model type, etc.) are set in a simple Python dictionary.
* **Automated Data Handling:** Includes a custom `Dataset` class that automatically handles image loading, transformations, and augmentations (e.g., resizing, random flips).
* **Multi-GPU Support:** Automatically leverages `torch.nn.DataParallel` for training on multiple GPUs to accelerate performance.
* **Best Model Saving:** Continuously tracks validation accuracy and saves only the best-performing model checkpoint.
* **Detailed Logging:** Provides clear console outputs for training progress, validation accuracy, and model summaries using `torchinfo`.

---

## 🚀 Projects Built with ClassiForge

This framework has been used to train several high-accuracy image classification models:

| Project Name | Model Architecture | Best Test Accuracy |
| :--- | :--- | :--- |
| **FlutterFrame** | ResNet-18 | **98.40%** |
| **Decksentience** | ResNet-18 | **97.36%** |
| **AthloScope** | ResNet-18 | **97.20%** |
| **BreedNova** | Vision Transformer | **96.86%** |

---

## 🛠️ Getting Started

Follow these instructions to set up the ClassiForge framework and run one of the example projects.

### 1. Installation

First, clone the repository and navigate into the project directory:
```bash
git clone [https://github.com/TheVictor777/ClassiForge.git](https://github.com/TheVictor777/ClassiForge.git)
cd ClassiForge
```

Next, create and activate a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Then, install the required dependencies using the clean requirements file:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

The datasets are not included in this repository.
1.  Create a folder named `Datasets` in the project's root directory.
2.  Download and unzip the required dataset into this folder. For example, for `AthloScope`:
    * **Download:** [100 Sports Image Classification on Kaggle](https://www.kaggle.com/datasets/gpiosenka/sports-classification)
    * **Folder Structure:** The final path should look like `Datasets/100 Sports Image Classification/train/...`

### 3. Running a Project

Each project is a simple Python script that imports and uses the ClassiForge framework. To start training, simply run the project's script.

For example, to train the **AthloScope** sports classifier:
```bash
python3 AthloScope.py
```

To train the **BreedNova** dog breed classifier (which uses a Vision Transformer):
```bash
python3 BreedNova.py
```

---

## 🔧 How It Works: A Look Inside

The power of ClassiForge comes from its modular design.

1.  **Project Scripts (e.g., `AthloScope.py`):** These are the entry points. They contain a single `config` dictionary with all the project-specific settings. This is where you define the model path, dataset folders, learning rate, and other hyperparameters.
2.  **The `ClassiForge.py` Framework:** This is the engine. The `start_training` function takes the `config` dictionary and handles the entire ML pipeline:
    * It validates that the dataset exists.
    * It creates the custom `classiforge_dataset` for training and testing.
    * It loads the specified pre-trained model (`ResNet` or `ViT`) and modifies its final layer for the correct number of classes.
    * It sets up the optimizer, loss function, and data loaders.
    * It runs the main training loop, calling the `train_step` and `test_step` functions for each epoch.
    * It saves the model checkpoint whenever a new best accuracy is achieved.

This separation of concerns makes it incredibly easy to start a new image classification project: just copy an existing project script, update the `config` dictionary with new paths and parameters, and run it.

from ClassiForge import start_training

# All settings are neatly organized in one place.
config = {  # Note: at the time of training, the current best testing accuracy is 97.71%
    "epochs": 25,
    "model_path": "Models/BreedNova_Model.pth",
    "train_folder": "Datasets/70 Dog Breeds-Image Data Set/train",
    "test_folder": "Datasets/70 Dog Breeds-Image Data Set/test",
    "learning_rate": 1e-6,
    "dataset_download_url": "https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set",

    # Showcasing compatibility with different architectures and hyperparameters.
    "model_type": "vit_b_16",
    "dropout": 0.6,
}

if __name__ == "__main__":
    # Clean function to start training.
    start_training(**config)

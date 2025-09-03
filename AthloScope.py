from ClassiForge import start_training

# All settings are neatly organized in one place.
config = {  # Note: at the time of training, the current best testing accuracy is 96.80%
    "epochs": 25,
    "model_path": "Models/AthloScope_Model.pth",
    "train_folder": "Datasets/100 Sports Image Classification/train",
    "test_folder": "Datasets/100 Sports Image Classification/test",
    "learning_rate": 1e-6,
    "dataset_download_url": "https://www.kaggle.com/datasets/gpiosenka/sports-classification",
}

if __name__ == "__main__":
    # Clean function to start training.
    start_training(**config)

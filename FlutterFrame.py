from ClassiForge import start_training

# All settings are neatly organized in one place.
config = {  # Note: at the time of training, the current best testing accuracy is 97.60%
    "epochs": 25,
    "model_path": "Models/FlutterFrame_Model.pth",
    "train_folder": "Datasets/Butterfly & Moths Image Classification 100 species/train",
    "test_folder": "Datasets/Butterfly & Moths Image Classification 100 species/test",
    "learning_rate": 1e-6,
    "dataset_download_url": "https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species",
}

if __name__ == "__main__":
    # Clean function to start training.
    start_training(**config)

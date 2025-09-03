from ClassiForge import start_training

# All settings are neatly organized in one place.
config = {  # Note: at the time of training, the current best testing accuracy is 96.98%
    "epochs": 25,
    "model_path": "Models/Decksentience_Model.pth",
    "train_folder": "Datasets/Cards Image Dataset-Classification/train",
    "test_folder": "Datasets/Cards Image Dataset-Classification/test",
    "learning_rate": 1e-6,
    "dataset_download_url": "https://www.kaggle.com/datasets/86dcbfae1396038cba359d58e258915afd32de7845fd29ef6a06158f80d3cce8",
}

if __name__ == "__main__":
    # Clean function to start training.
    start_training(**config)

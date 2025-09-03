"""
This program creates a custom framework and pipeline that are used in various AI image classification projects.
It is designed to be flexible and efficient, allowing for easy integration of different models and datasets.
"""
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import torchinfo
import torch
import tqdm
import os

def load_model(model: torch.nn.Module, model_path: str, verbose: bool = True):
    """
    Load the model state from the specified path.
    If the model is not found, it will print an error message.

    Args:
        model (torch.nn.Module): The model to load the state into.
        model_path (str): The path to the model file.
        verbose (bool): If True, prints messages about the loading process.
    Returns:
        torch.nn.Module: The model with loaded state.
    """

    # Handle case where model file does not exist.
    if not os.path.exists(model_path):
        if verbose:
            print("\033[91m" + f"[CLASSIFORGE] Model file not found at {model_path}" + "\033[0m")  # RED
        return model, 0.0

    # Attempt to load the model state
    try:
        loaded_data = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(loaded_data['model_state_dict'])  # Load the model state
        highest_accuracy = loaded_data.get('highest_accuracy', 0.0)  # Default to 0.0 if not found
        if verbose:
            print("\033[92m" + f"[CLASSIFORGE] Model loaded successfully from {model_path} with accuracy {highest_accuracy:.2f}%" + "\033[0m")  # GREEN

    # Catch any exceptions during loading
    except Exception as e:
        if verbose:
            print("\033[91m" + f"[CLASSIFORGE] Error loading model: {e}" + "\033[0m")  # RED
        highest_accuracy = 0.0

    return model, highest_accuracy

def model_to_multi_gpu(model: torch.nn.Module, verbose: bool = True):
    """
    Convert the model to use multiple GPUs if available.
    This function uses DataParallel instead of DistributedDataParallel for simplicity.

    Args:
        model (torch.nn.Module): The model to convert.
        verbose (bool): If True, prints messages about the conversion process.
    Returns:
        torch.nn.Module: The model wrapped for multi-GPU training.
    """
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        if verbose:
            print("\033[95m" + f"[CLASSIFORGE] Model set to use {torch.cuda.device_count()} GPUs with DataParallel." + "\033[0m")  # PINK

    return model

def set_random_seed(Seed: int):
    """
    Set the random seed for reproducibility.
    Args:
        Seed (int): The seed value to set.
    """
    torch.manual_seed(Seed)
    np.random.seed(Seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Seed)
    print("\033[95m" + f"[CLASSIFORGE] Random seed set to {Seed}" + "\033[0m")  # PINK

class classiforge_dataset(torch.utils.data.Dataset):
    """
    Custom dataset for loading images from a directory structure where each subdirectory represents a class.

    Args:
        main_directory (str): The main directory containing class subdirectories.
    Returns:
        torch.utils.data.Dataset: A dataset object for loading images and their corresponding labels.
    """
    def __init__(self, main_directory: str):
        self.main_directory = main_directory
        self.class_folders = sorted([directory for directory in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, directory))])  # Sort class folders.
        self.image_paths = []
        self.labels = []

        for index, class_name in enumerate(self.class_folders):
            ClassPaths = os.path.join(main_directory, class_name)
            for ImageFiles in os.listdir(ClassPaths):
                if ImageFiles.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(ClassPaths, ImageFiles))
                    self.labels.append(index)

        self.DatasetTransform = transforms.Compose([
            transforms.Resize((224, 224)),  # Scale.
            transforms.RandomHorizontalFlip(),  # Horizontal flip.
            transforms.RandomVerticalFlip(),  # Vertical flip.
            transforms.ToTensor(),

            # Normalize for resnet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        Photo = Image.open(self.image_paths[index]).convert("RGB")
        Photo = self.DatasetTransform(Photo)
        Label = torch.tensor(self.labels[index], dtype=torch.long)
        return Photo, Label

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The data loader for training data.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to run the training on.
    """
    model.train()
    running_loss = 0.0
    for index, (input_tensor, labels) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
        input_tensor, labels = input_tensor.to(device), labels.to(device)
        optimizer.zero_grad()
        output_tensor = model(input_tensor)
        loss = criterion(output_tensor, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"[CLASSIFORGE] Epoch training loop finished with training loss: {running_loss/len(dataloader):.7f}")

def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    """
    Test the model and calculate accuracy.

    Args:
        model (torch.nn.Module): The model to test.
        dataloader (torch.utils.data.DataLoader): The data loader for testing data.
        device (torch.device): The device to run the testing on.
    Returns:
        float: The accuracy of the model on the test dataset.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for index, (input_tensor, labels) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing"):
            input_tensor, labels = input_tensor.to(device), labels.to(device)
            output_tensor = model(input_tensor)
            _, preds = torch.max(output_tensor, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct/total*100
    print(f"[CLASSIFORGE] Epoch testing loop finished with testing accuracy: {accuracy:.2f}%")
    return accuracy

def validate_dataset(train_folder: str, test_folder: str, dataset_download_url: str):
    """
    Validate the dataset by checking if the training and testing folders exist and contain data files.

    Args:
        train_folder (str): The path to the training folder.
        test_folder (str): The path to the testing folder.
        dataset_download_url (str): URL to refer users to download the dataset if folders are missing.
    """

    # Ensure that a dataset link has been provided if folders are missing.
    if dataset_download_url is None:
        raise ValueError("Dataset download URL must be provided to validate dataset folders.")

    # Check training folder validity.
    if not os.path.isdir(train_folder):
        print("\033[91m" + f"Training folder '{train_folder}' does not exist." + "\033[0m")  # RED
        print("\033[93m" + f"Please download the dataset from {dataset_download_url} before trying again." + "\033[0m")  # YELLOW
        quit()

    # Check testing folder validity.
    if not os.path.isdir(test_folder):
        print("\033[91m" + f"Testing folder '{test_folder}' does not exist." + "\033[0m")  # RED
        print("\033[93m" + f"Please download the dataset from {dataset_download_url} before trying again." + "\033[0m")  # YELLOW
        quit()

    # Check if there are files in the training folder.
    if len(os.listdir(train_folder)) == 0: raise ValueError(f"Training folder '{train_folder}' is empty.")

    # Check if there are files in the testing folder.
    if len(os.listdir(test_folder)) == 0: raise ValueError(f"Testing folder '{test_folder}' is empty.")

    print("\033[92m" + "[CLASSIFORGE] Dataset validation passed." + "\033[0m")  # GREEN

def start_training(epochs: int, model_path: str, train_folder: str, test_folder: str, learning_rate: float = 1e-4, batch_size: int = 64, dataset_download_url: str = None, model_type: str = "resnet18", dropout: float = 0.5, weight_decay: float = 1e-6, num_workers = max(1, os.cpu_count()-1)):
    """
    Start the training process for the model.

    Args:
        epochs (int): The number of epochs to train the model.
        model_path (str): The path where the model will be saved.
        train_folder (str): The path to the training dataset folder.
        test_folder (str): The path to the testing dataset folder.
        learning_rate (float): The learning rate for the optimizer.
        batch_size (int): The batch size for training and testing.
        dataset_download_url (str): Optional URL for downloading the dataset if it does not exist.
        model_type (str): The pre-trained model to use, e.g., "resnet18" or "vit_b_16".
        dropout (float): dropout rate for the model's fully connected layer.
        weight_decay (float): Weight decay for the optimizer.
        num_workers (int): Number of workers for data loading, set to one less than the total CPU cores available.
    """

    # Set random seed.
    set_random_seed(42)

    # Immediately check dataset validity.
    validate_dataset(train_folder, test_folder, dataset_download_url)

    # Set up distributed training if applicable.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\033[95m" + f"[CLASSIFORGE] Using device: {device}" + "\033[0m")  # PINK

    # Set up datasets.
    train_dataset = classiforge_dataset(train_folder)
    test_dataset = classiforge_dataset(test_folder)

    # Setup model.
    if model_type == "resnet18":
        backbone = models.resnet18(weights='IMAGENET1K_V1')
        backbone.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(backbone.fc.in_features, len(train_dataset.class_folders)))
    elif model_type == "vit_b_16":
        backbone = models.vit_b_16(weights='IMAGENET1K_V1')
        backbone.heads = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(backbone.heads.head.in_features, len(train_dataset.class_folders)))
    else:
        raise ValueError(f"Unsupported model: {model_type}.")

    model = backbone.to(device)
    model, highest_accuracy = load_model(model, model_path)
    model = model_to_multi_gpu(model)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=1)

    # Get model summary only on the main process to avoid clutter.
    print("\033[94m" + "[CLASSIFORGE] Model Summary:" + "\033[0m")  # BLUE
    torchinfo.summary(model, input_size=(1, 3, 224, 224), device=device.type)  # A single batched RGB image of size 224x224.

    # Training Parameters.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training Loop.
    for epoch in range(epochs):
        print("\033[95m" + f"[CLASSIFORGE] Epoch {epoch+1}/{epochs}" + "\033[0m")  # PINK

        train_step(model, train_loader, criterion, optimizer, device)
        accuracy = test_step(model, test_loader, device)

        # Save the model only from the main process.
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            # Correctly save the model's state_dict when wrapped.
            save_data = {
                'model_state_dict': model.state_dict() if not isinstance(model, torch.nn.DataParallel) else model.module.state_dict(),
                'highest_accuracy': highest_accuracy
            }
            torch.save(save_data, model_path)
            print("\033[92m" + "[CLASSIFORGE] Best model saved!" + "\033[0m")
        print()  # Separator for clarity between epochs.

if __name__ == "__main__":
    print("This is a utility module and is not meant to be run directly.")
    print("Please import this module in your project to use the ClassiForge framework.")
    # Example usage:
    # start_training(epochs=10, model_path='model.pth', train_folder='path/to/train', test_folder='path/to/test')

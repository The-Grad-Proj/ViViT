import sys
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import wandb
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from dataloaders.dataset import CustomDataset
from models.ViViT import ViViT

# Hyperparameters
PATCH_SIZE = 16
BATCH_SIZE = 32
FRAMES_NUM = 16
STRIDE = 4
LR = 0.001
IMAGE_SIZE = 224
EPOCHS_NUM = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "saved_models/vivit_model.pth"


# Define Data Directories
root_dir = '/home/ibraa04/grad_project/udacity/output'
csv_path = root_dir + '/CH2_final_evaluation.csv'

criterion =  torch.nn.MSELoss()
criterion.to(DEVICE)


# Define Data Transformations
transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

def get_data_loaders():
    dataset_instance = CustomDataset(csv_file=csv_path, root_dir=root_dir, T=FRAMES_NUM, stride=STRIDE, transform=transform)
    dataset_size = len(dataset_instance)
    indices = list(range(dataset_size))
    
    eval_sampler = SubsetRandomSampler(indices=indices)    
    eval_loader = DataLoader(dataset_instance, batch_size=BATCH_SIZE, sampler=eval_sampler)
    
    return eval_loader


def load_model():
    model = ViViT(image_size = IMAGE_SIZE, patch_size = PATCH_SIZE, num_classes = 1, num_frames = FRAMES_NUM).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Loaded existing model from saved_models/")
    else:
        print("No saved model found. Instantiating a new model.")
    return model

def main():
    validation_loader = get_data_loaders()

    model = load_model()
    model.to(DEVICE)
    model.eval()

    validation_loss_angle = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(validation_loader)):
            inputs, targets = data
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            validation_loss_angle += torch.sqrt(loss + 1e-6).item()
        avg_validation_loss_angle = validation_loss_angle / len(validation_loader)
        print(f"Validation Loss: {avg_validation_loss_angle}")

if __name__ == "__main__":
    main()
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
sys.path.append("D:\AI\Graduation Project\ViViT")
from dataloaders.dataset import CustomDataset
from models.ViViT import ViViT

# Fix numpy and pytorch seed
np.random.seed(42)
torch.manual_seed(42)

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
# root_dir = '/home/ibraa04/grad_project/udacity/output'
# csv_path = root_dir + '/interpolated.csv'

root_dir = r"D:\AI\Graduation Project\Udacity Dataset"
csv_path = r"D:\AI\Graduation Project\Udacity Dataset\interpolated.csv"

# Define Data Transformations
transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])


def get_data_loaders():
    dataset_instance = CustomDataset(csv_file=csv_path, root_dir=root_dir, T=FRAMES_NUM, stride=STRIDE, transform=transform)
    dataset_size = len(dataset_instance)
    indices = list(range(dataset_size))
    
    np.random.shuffle(indices)
    split = int(0.9 * dataset_size)  # 90% train, 10% validation
    train_indices, val_indices = indices[:split], indices[split:]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset_instance, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(dataset_instance, batch_size=BATCH_SIZE, sampler=val_sampler)
    
    return train_loader, val_loader


def load_model():
    model = ViViT(image_size = IMAGE_SIZE, patch_size = PATCH_SIZE, num_classes = 2, num_frames = FRAMES_NUM).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Loaded existing model from saved_models/")
    else:
        print("No saved model found. Instantiating a new model.")
    return model


def train(model, train_loader, angle_criterion, speed_criterion, optimizer):
    """Train the model and log steering/speed losses using WandB."""
    model.train()
    total_loss = 0
    total_steering_loss = 0
    total_speed_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        frames, steering_angle, speed = batch
        frames, steering_angle, speed = frames.to(DEVICE), steering_angle.to(DEVICE), speed.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(frames)

        # Split outputs and targets
        pred_steering, pred_speed = outputs[:, 0], outputs[:, 1]

        # Compute individual losses
        loss_steering = torch.sqrt(angle_criterion(pred_steering, steering_angle) + 1e-6)
        loss_speed = speed_criterion(pred_speed, speed)

        # Total loss is a weighted sum (can be adjusted)
        loss = loss_steering + loss_speed
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_steering_loss += loss_steering.item()
        total_speed_loss += loss_speed.item()

    avg_train_loss = total_loss / len(train_loader)
    avg_steering_loss = total_steering_loss / len(train_loader)
    avg_speed_loss = total_speed_loss / len(train_loader)

    print(f"Train Loss: {avg_train_loss:.4f} | Steering RMSE: {avg_steering_loss:.4f} | Speed SmoothL1: {avg_speed_loss:.4f}")
    
    # wandb.log({
    #     "Train Loss": avg_train_loss,
    #     "Train Steering Loss": avg_steering_loss,
    #     "Train Speed Loss": avg_speed_loss
    # })

    return avg_train_loss


def validate(model, val_loader, angle_criterion, speed_criterion):
    """Validate the model and log steering/speed losses using WandB."""
    model.eval()
    total_loss = 0
    total_steering_loss = 0
    total_speed_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            frames, steering_angle, speed = batch
            frames, steering_angle, speed = frames.to(DEVICE), steering_angle.to(DEVICE), speed.to(DEVICE)

            outputs = model(frames)

            pred_steering, pred_speed = outputs[:, 0], outputs[:, 1]

            loss_steering = torch.sqrt(angle_criterion(pred_steering, steering_angle) + 1e-6)
            loss_speed = speed_criterion(pred_speed, speed)

            loss = loss_steering + loss_speed

            total_loss += loss.item()
            total_steering_loss += loss_steering.item()
            total_speed_loss += loss_speed.item()

    avg_val_loss = total_loss / len(val_loader)
    avg_steering_loss = total_steering_loss / len(val_loader)
    avg_speed_loss = total_speed_loss / len(val_loader)

    print(f"Validation Loss: {avg_val_loss:.4f} | Steering RMSE: {avg_steering_loss:.4f} | Speed SmoothL1: {avg_speed_loss:.4f}")

    # wandb.log({
    #     "Validation Loss": avg_val_loss,
    #     "Validation Steering Loss": avg_steering_loss,
    #     "Validation Speed Loss": avg_speed_loss
    # })

    return avg_val_loss


def main():
    # Initialize WandB
    # wandb.init(project="vivit-training", config={
    #     "batch_size": BATCH_SIZE,
    #     "frames_num": FRAMES_NUM,
    #     "stride": STRIDE,
    #     "learning_rate": LR,
    #     "epochs": EPOCHS_NUM,
    # })

    train_loader, val_loader = get_data_loaders()
    model = load_model()

    angle_criterion = torch.nn.MSELoss()
    angle_criterion.to(DEVICE)

    speed_criterion = torch.nn.SmoothL1Loss()
    speed_criterion.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS_NUM):
        print(f"Epoch [{epoch+1}/{EPOCHS_NUM}]")
        train_loss = train(model, train_loader, angle_criterion, speed_criterion, optimizer)
        val_loss = validate(model, val_loader, angle_criterion, speed_criterion)

        print(f"Epoch [{epoch+1}/{EPOCHS_NUM}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved in saved_models/")

    # Finish WandB logging
    # wandb.finish()


if __name__ == "__main__":
    main()

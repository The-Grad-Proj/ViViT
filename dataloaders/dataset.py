from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage import io
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, T=16, stride=1, transform=None):
        """
        Args:
            csv_file (str): Path to CSV file with filenames and steering angles.
            root_dir (str): Directory with all images.
            T (int): Number of frames per sequence.
            stride (int): Step size for overlapping sampling.
            transform (callable, optional): Transform to apply to images.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.csv_file = self.csv_file[self.csv_file['frame_id'] == 'center_camera']  # Select center camera only
        self.root_dir = root_dir
        self.T = T
        self.stride = stride
        self.transform = transform
        self.samples = self._generate_samples()

    def _generate_samples(self):
        """Generate indices for overlapping sequences."""
        samples = []
        num_frames = len(self.csv_file)
        
        for i in range(0, num_frames - self.T, self.stride):  
            samples.append(i)  # Store starting index of each sequence
        
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns a sequence of T frames + last frame's steering angle"""
        start_idx = self.samples[idx]
        frame_sequence = []
        
        for i in range(start_idx, start_idx + self.T):
            img_path = os.path.join(self.root_dir, self.csv_file['filename'].iloc[i])
            image = io.imread(img_path)
            image = Image.fromarray(image)

            if self.transform:
                image = self.transform(image)

            frame_sequence.append(image)

        # Convert list of T frames into a (T, C, H, W) tensor
        frame_sequence = torch.stack(frame_sequence, dim=0)

        # Steering angle of the last frame in the sequence
        steering_angle = self.csv_file['angle'].iloc[start_idx + self.T - 1]

        # Speed of the last frame in the sequence
        speed = self.csv_file['speed'].iloc[start_idx + self.T - 1]

        return frame_sequence, torch.tensor(steering_angle, dtype=torch.float32), torch.tensor(speed, dtype=torch.float32)

# ------------------ TEST THE DATASET ------------------


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    root_dir = r"D:\AI\Graduation Project\Udacity Dataset"
    csv_path = r"D:\AI\Graduation Project\Udacity Dataset\interpolated.csv"

    dataset = CustomDataset(csv_file=csv_path, root_dir=root_dir, T=16, stride=4, transform=transform)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Set batch_size=1 for single samples

    # Create an iterator
    data_iter = iter(dataloader)

    for _ in range(5):

        # Get the first sample using next()
        frames, steering, speed = next(data_iter)

        print(f"Steering angle: {steering.item()}")
        print(f"Speed: {speed.item()}")
        print(f"Frames shape: {frames.shape}")  # Should be (1, T, C, H, W) because of batch size

        # Display the first frame of the sequence
        image = frames[0, 0].permute(1, 2, 0).numpy()  # [batch, T, C, H, W] → Select first batch and first frame
        plt.imshow(image)
        plt.axis("off")
        plt.show()

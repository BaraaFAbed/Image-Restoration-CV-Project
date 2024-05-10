import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch.utils.data import random_split

class GoProDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Transform to be applied on the input images.
            target_transform (callable, optional): Transform to be applied on the target images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []

        # Iterate over each sequence folder
        for sequence in os.listdir(root_dir):
            sequence_dir = os.path.join(root_dir, sequence)
            blur_dir = os.path.join(sequence_dir, 'blur')
            sharp_dir = os.path.join(sequence_dir, 'sharp')

            # List and sort blur and sharp images to ensure they are aligned
            blur_images = sorted([os.path.join(blur_dir, f) for f in os.listdir(blur_dir) if f.endswith('.png')])
            sharp_images = sorted([os.path.join(sharp_dir, f) for f in os.listdir(sharp_dir) if f.endswith('.png')])

            # Pair each blur image with its corresponding sharp image
            self.samples.extend(zip(blur_images, sharp_images))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.samples[idx]
        blur_image = Image.open(blur_path)
        sharp_image = Image.open(sharp_path)

        if self.transform:
            blur_image = self.transform(blur_image)
        if self.target_transform:
            sharp_image = self.target_transform(sharp_image)

        return blur_image, sharp_image


def getDataset():
    # Define transformations if needed
    transform = transforms.Compose([
        transforms.CenterCrop((540,960)),
        transforms.ToTensor(),
    ])
    target_transform = transforms.Compose([
        transforms.CenterCrop((540,960)),
        transforms.ToTensor(),
    ])
    
    # Create dataset
    train_dataset = GoProDataset(root_dir='GoPro/train', transform=transform, target_transform=target_transform)
    test_dataset = GoProDataset(root_dir='GoPro/test', transform=transform, target_transform=target_transform)
    
    # Define the size of the validation set
    val_size = int(0.2 * len(train_dataset))  # 20% of the training dataset for validation
    
    # Split train_dataset into train and validation sets
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    return train_dataset, val_dataset, test_dataset
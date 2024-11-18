import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os
from torch.utils.data import ConcatDataset
import random
import numpy as np

def create_augmented_dataset(original_dataset, num_augmented=0.05):
    # Calculate how many augmented samples to create (5% of original)
    num_samples = int(len(original_dataset) * num_augmented)
    
    # Create augmentation transforms
    shear_transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, shear=20),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    rotation_transform = transforms.Compose([
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Create augmented datasets
    shear_dataset = datasets.MNIST('data', train=original_dataset.train, 
                                 download=True, transform=shear_transform)
    rotation_dataset = datasets.MNIST('data', train=original_dataset.train, 
                                    download=True, transform=rotation_transform)
    
    # Randomly select indices for augmented samples
    indices = random.sample(range(len(original_dataset)), num_samples)
    
    # Create subset datasets
    shear_subset = torch.utils.data.Subset(shear_dataset, indices[:num_samples//2])
    rotation_subset = torch.utils.data.Subset(rotation_dataset, indices[num_samples//2:])
    
    # Combine original and augmented datasets
    combined_dataset = ConcatDataset([original_dataset, shear_subset, rotation_subset])
    
    return combined_dataset

def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Base transform for original images
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load original dataset
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=base_transform)
    
    # Create augmented training dataset
    augmented_train_dataset = create_augmented_dataset(train_dataset)
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(augmented_train_dataset, 
                                             batch_size=64, 
                                             shuffle=True)
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for one epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(model.state_dict(), f'saved_models/model_{timestamp}.pth')
    
if __name__ == "__main__":
    train() 
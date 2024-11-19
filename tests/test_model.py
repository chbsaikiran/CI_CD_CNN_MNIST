import torch
import pytest
import sys
import os
import glob
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from PIL import Image
import random
from skimage.metrics import structural_similarity as ssim
import numpy as np
from torchvision.utils import make_grid

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.network import SimpleCNN
from torchvision import datasets, transforms

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    """Test if model has less than 25000 parameters"""
    model = SimpleCNN()
    param_count = count_parameters(model)
    print(f"\nTest 1: Parameter Count Check")
    print(f"Total parameters: {param_count}")
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"
    print("✓ PASSED: Model parameter count is within limit")

def test_input_shape():
    """Test if model accepts 28x28 input"""
    print(f"\nTest 2: Input Shape Check")
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
        print("✓ PASSED: Model successfully processes 28x28 input")
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        assert False, "Model failed to process 28x28 input"

def test_output_shape():
    """Test if model outputs 10 classes"""
    print(f"\nTest 3: Output Shape Check")
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    assert output.shape[1] == 10, f"Output shape is {output.shape[1]}, should be 10"
    print("✓ PASSED: Model outputs correct number of classes (10)")

def test_model_accuracy():
    """Test if model achieves >95% accuracy"""
    print(f"\nTest 4: Model Accuracy Check")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    # Load the latest model
    import glob
    import os
    model_files = glob.glob('saved_models/model_*.pth')
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Loading model: {latest_model}")
    model.load_state_dict(torch.load(latest_model))
    
    # Test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Model accuracy: {accuracy:.2f}%")
    assert accuracy > 95, f"Accuracy is {accuracy:.2f}%, should be > 95%"
    print("✓ PASSED: Model achieves required accuracy")

def visualize_all_transformations(original_images, sheared_images, rotated_images, scaled_images, save_path):
    """Helper function to visualize and save original vs all transformed images"""
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle("Original vs All Transformations")
    
    # Show images for each row
    for i in range(5):
        # Original images
        axes[0, i].imshow(original_images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('Original' if i == 0 else '')
        
        # Sheared images
        axes[1, i].imshow(sheared_images[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Sheared' if i == 0 else '')
        
        # Rotated images
        axes[2, i].imshow(rotated_images[i].squeeze(), cmap='gray')
        axes[2, i].axis('off')
        axes[2, i].set_title('Rotated' if i == 0 else '')
        
        # Scaled images
        axes[3, i].imshow(scaled_images[i].squeeze(), cmap='gray')
        axes[3, i].axis('off')
        axes[3, i].set_title('Scaled' if i == 0 else '')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_combined_transformations():
    """Test model performance on combined transformations (shear + rotation + scale)"""
    print(f"\nTest 5: Combined Transformations Performance Check")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    # Load the latest model
    model_files = glob.glob('saved_models/model_*.pth')
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Loading model: {latest_model}")
    model.load_state_dict(torch.load(latest_model))
    
    # Create transforms
    base_transform = transforms.ToTensor()
    
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
    
    scale_transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load test dataset
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=base_transform)
    
    # Create visualization samples
    sample_images = []
    sheared_samples = []
    rotated_samples = []
    scaled_samples = []
    
    # Get 5 sample images for visualization
    for i in range(5):
        img_pil, _ = test_dataset.data[i], test_dataset.targets[i]
        img_pil = Image.fromarray(img_pil.numpy())
        
        # Original
        sample_images.append(base_transform(img_pil))
        
        # Transformations (without normalization for visualization)
        sheared_samples.append(transforms.Compose([
            transforms.RandomAffine(degrees=0, shear=20),
            transforms.ToTensor()
        ])(img_pil))
        
        rotated_samples.append(transforms.Compose([
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor()
        ])(img_pil))
        
        scaled_samples.append(transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
            transforms.ToTensor()
        ])(img_pil))
    
    # Save visualization
    visualize_all_transformations(sample_images, sheared_samples, rotated_samples, scaled_samples,
                                "combined_transformations_samples.png")
    print("✓ Generated visualization of all transformations (saved as 'combined_transformations_samples.png')")
    
    # Create combined test dataset with different transformations
    total_samples = len(test_dataset)
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    # Split indices into three parts
    split1 = total_samples // 3
    split2 = 2 * total_samples // 3
    
    # Create subsets with different transformations
    shear_dataset = datasets.MNIST('data', train=False, download=True, transform=shear_transform)
    rotation_dataset = datasets.MNIST('data', train=False, download=True, transform=rotation_transform)
    scale_dataset = datasets.MNIST('data', train=False, download=True, transform=scale_transform)
    
    shear_subset = torch.utils.data.Subset(shear_dataset, indices[:split1])
    rotation_subset = torch.utils.data.Subset(rotation_dataset, indices[split1:split2])
    scale_subset = torch.utils.data.Subset(scale_dataset, indices[split2:])
    
    # Combine all transformed subsets
    combined_dataset = torch.utils.data.ConcatDataset([shear_subset, rotation_subset, scale_subset])
    combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=1000)
    
    # Test accuracy on combined transformed dataset
    model.eval()
    correct = 0
    total = 0
    
    print("\nTesting on combined transformed dataset (1/3 shear, 1/3 rotation, 1/3 scale)...")
    
    with torch.no_grad():
        for data, target in combined_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Model accuracy on combined transformed images: {accuracy:.2f}%")
    assert accuracy > 90, f"Accuracy on combined transformed images is {accuracy:.2f}%, should be > 90%"
    print("✓ PASSED: Model achieves required accuracy on combined transformed images")

def test_data_integrity():
    """Test if augmented images maintain data integrity"""
    print(f"\nTest 6: Data Integrity Check")
    
    # Create transforms
    shear_transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, shear=20),
        transforms.ToTensor(),
    ])
    
    rotation_transform = transforms.Compose([
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
    ])
    
    scale_transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ])
    
    # Load a batch of test images
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    batch_size = 10
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    original_batch, _ = next(iter(test_loader))
    
    # Apply transformations
    img_pil = transforms.ToPILImage()(original_batch[0])
    sheared = shear_transform(img_pil)
    rotated = rotation_transform(img_pil)
    scaled = scale_transform(img_pil)
    
    # Test 1: Check image size
    print("\nChecking image dimensions...")
    assert sheared.shape == (1, 28, 28), f"Sheared image size is {sheared.shape}, expected (1, 28, 28)"
    assert rotated.shape == (1, 28, 28), f"Rotated image size is {rotated.shape}, expected (1, 28, 28)"
    assert scaled.shape == (1, 28, 28), f"Scaled image size is {scaled.shape}, expected (1, 28, 28)"
    print("✓ PASSED: All augmented images maintain correct dimensions")
    
    # Test 2: Check pixel value range
    print("\nChecking pixel value ranges...")
    for img, name in [(sheared, "Sheared"), (rotated, "Rotated"), (scaled, "Scaled")]:
        min_val, max_val = img.min().item(), img.max().item()
        assert 0 <= min_val <= 1, f"{name} image has min value {min_val}, should be between 0 and 1"
        assert 0 <= max_val <= 1, f"{name} image has max value {max_val}, should be between 0 and 1"
    print("✓ PASSED: All augmented images maintain valid pixel value ranges")
    
    # Test 3: Check for extreme distortions using SSIM
    print("\nChecking for extreme distortions...")
    original_np = original_batch[0].squeeze().numpy()
    
    for img, name in [(sheared, "Sheared"), (rotated, "Rotated"), (scaled, "Scaled")]:
        img_np = img.squeeze().numpy()
        similarity = ssim(original_np, img_np, data_range=1.0)
        assert similarity > 0.3, f"{name} image has SSIM {similarity}, should be > 0.3"
        print(f"{name} image SSIM: {similarity:.3f}")
    
    print("✓ PASSED: All augmented images maintain reasonable similarity to originals")

def test_augmentation_diversity():
    """Test if augmentations produce diverse variations"""
    print(f"\nTest 7: Augmentation Diversity Check")
    
    # Create transforms
    transforms_list = {
        'shear': transforms.Compose([
            transforms.RandomAffine(degrees=0, shear=20),
            transforms.ToTensor(),
        ]),
        'rotation': transforms.Compose([
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
        ]),
        'scale': transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
            transforms.ToTensor(),
        ])
    }
    
    # Load a single test image
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    original_image = test_dataset[0][0]
    img_pil = transforms.ToPILImage()(original_image)
    
    # Generate multiple versions of each augmentation
    n_versions = 5
    augmented_versions = {name: [] for name in transforms_list.keys()}
    
    print("\nGenerating multiple versions of each augmentation...")
    for name, transform in transforms_list.items():
        for _ in range(n_versions):
            aug_img = transform(img_pil)
            augmented_versions[name].append(aug_img)
    
    # Test 1: Check pixel variance across versions
    print("\nChecking pixel variance across augmented versions...")
    min_variance_threshold = 1e-4
    
    for name, versions in augmented_versions.items():
        # Stack tensors and calculate variance
        stacked = torch.stack(versions)
        pixel_variance = torch.var(stacked, dim=0).mean().item()
        
        print(f"{name.capitalize()} augmentation variance: {pixel_variance:.6f}")
        assert pixel_variance > min_variance_threshold, \
            f"{name} augmentation has low variance ({pixel_variance:.6f}), should be > {min_variance_threshold}"
    
    print("✓ PASSED: All augmentations show sufficient variance")
    
    # Test 2: Check SSIM between pairs
    print("\nChecking structural similarity between augmented versions...")
    max_similarity_threshold = 0.98
    
    for name, versions in augmented_versions.items():
        similarities = []
        for i in range(n_versions):
            for j in range(i + 1, n_versions):
                img1 = versions[i].squeeze().numpy()
                img2 = versions[j].squeeze().numpy()
                similarity = ssim(img1, img2, data_range=1.0)
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities)
        print(f"{name.capitalize()} average SSIM between versions: {avg_similarity:.3f}")
        assert avg_similarity < max_similarity_threshold, \
            f"{name} augmentation versions are too similar (SSIM: {avg_similarity:.3f}), should be < {max_similarity_threshold}"
    
    print("✓ PASSED: All augmentations produce sufficiently diverse versions")
    
    # Visualize diversity
    fig, axes = plt.subplots(len(transforms_list), n_versions + 1, figsize=(15, 8))
    fig.suptitle("Augmentation Diversity Visualization")
    
    for i, (name, versions) in enumerate(augmented_versions.items()):
        # Show original
        axes[i, 0].imshow(original_image.squeeze(), cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Original' if i == 0 else '')
        
        # Show augmented versions
        for j, img in enumerate(versions):
            axes[i, j + 1].imshow(img.squeeze(), cmap='gray')
            axes[i, j + 1].axis('off')
            if i == 0:
                axes[i, j + 1].set_title(f'Version {j + 1}')
        
        axes[i, 0].set_ylabel(name.capitalize())
    
    plt.tight_layout()
    plt.savefig("augmentation_diversity.png")
    plt.close()
    print("\n✓ Generated visualization of augmentation diversity (saved as 'augmentation_diversity.png')")
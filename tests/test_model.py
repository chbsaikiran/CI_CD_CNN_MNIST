import torch
import pytest
import sys
import os

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
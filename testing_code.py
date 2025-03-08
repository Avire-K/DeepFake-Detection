import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os

def evaluate_model():
    # Define data transforms for the test data
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the test dataset
    base_dir = os.getcwd()  # Base directory
    test_dir = os.path.join(base_dir, 'test')  # Path to your test folder
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load the saved model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)  # Initialize the model without pretrained weights
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(test_dataset.classes))  # Update for the number of classes
    model_path = os.path.join(base_dir, 'current_best_model.pt')  # Path to the .pth file

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract the state_dict from the checkpoint (it should be under 'state_dict' in the checkpoint)
    model.load_state_dict(checkpoint['state_dict'], strict=False)  # Use strict=False to ignore unexpected keys

    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Evaluate the model on the test dataset
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get predicted class indices
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    # Calculate accuracy
    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

if __name__ == "__main__":
    evaluate_model()

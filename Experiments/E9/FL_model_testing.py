"""
Federated Learning Model Testing
This module provides functions to evaluate the performance of federated learning models.
"""
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('TkAgg')  # Use non-interactive backend

# Suppress TensorFlow warnings
if os.name == 'posix':
    os.dup2(os.open('/dev/null', os.O_WRONLY), 2)
    
# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    """
    Convolutional Neural Network for CIFAR-10 classification.
    Its architecture consists of 2 convolutional layers with batch normalization,
    followed by 2 fully connected layers with dropout.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  #  BatchNorm added
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # BatchNorm added
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)  #  Dropout added
        self.fc2 = nn.Linear(256, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  #BatchNorm applied
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # BatchNorm applied
        x = x.view(-1, 64 * 8 * 8)  # Flatten for FC layers
        x = self.dropout(self.relu(self.fc1(x)))  #  Dropout before fc2
        x = self.fc2(x)  #  No softmax
        return x

# Load CIFAR-10 Test Data
def load_test_data(batch_size=128):
    """
    Load CIFAR-10 test dataset.
    
    Args:
        batch_size: Batch size for DataLoader
        
    Returns:
        DataLoader: Test data loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    # Verification print to check test set size
    #print(f"Test set size: {len(testset)}")
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return test_loader

def evaluate_model(model, test_loader, device):
    """
    Evaluate model performance on a test dataset.
    
    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        device: Computing device (CPU/GPU)
        
    Returns:
        tuple: (accuracy, average loss, confusion matrix)
    """
    model.to(device)
    model.eval()
    
    correct, total, test_loss = 0, 0, 0.0
    all_preds = []
    all_labels = []

    # Define class weights (adjust based on dataset imbalance if needed)
    loss_fn = torch.nn.CrossEntropyLoss()  # Use default loss without normalization

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
           

            # Compute loss using weighted CrossEntropyLoss
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

            # Debug print to inspect raw outputs
            #print(f"Sample outputs (logits): {outputs[:2]}")  # Print logits for first 2 samples
            _, predicted = torch.max(outputs, 1)
            # Debug print to verify predictions vs. labels
            #print(f"Sample predictions: {predicted[:10]}, Sample labels: {labels[:10]}")
            correct += (predicted == labels).sum().item()
            #correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    avg_loss = test_loss / len(test_loader)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return accuracy, avg_loss, conf_matrix


def plot_confusion_matrix(conf_matrix):
    """Plots the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show(block=True)  


def load_model():
    """
    Load the trained federated learning model.
    
    Returns:
        nn.Module: Loaded model
    """
    from cifar10_partition import CNN  # Import model class
    model = CNN().to(device) # Initialize the model architecture
    model.load_state_dict(torch.load("global_model.pth", map_location=device))  # Load weights
    model.eval()
    return model


if __name__ == "__main__":
    print("⚙ ⚙ ⚙ Loading test data...")
    test_loader = load_test_data()

    print("⚙ ⚙ ⚙ Loading trained global model...")
    model = load_model()
    
    print("⚙ ⚙ ⚙ Evaluating model on test set...")
    accuracy, avg_loss, conf_matrix = evaluate_model(model, test_loader, device)
    print(f"Final Test Accuracy: {accuracy:.4f}, Test Loss: {avg_loss:.4f}")
    
    print("⚙ ⚙ ⚙ Generating confusion matrix...")
    plot_confusion_matrix(conf_matrix)  
    
    print("✔ Model evaluation completed!")
    
    
    # Evaluate the model
    accuracy, avg_loss, conf_matrix = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {avg_loss:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix)
    


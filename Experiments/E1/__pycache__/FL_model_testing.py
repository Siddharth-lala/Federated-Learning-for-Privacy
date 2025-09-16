import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10 Test Data
def load_test_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return test_loader

def evaluate_model(model, test_loader, device):
    """Evaluates the trained FL model on the CIFAR-10 test set."""
    model.to(device)
    model.eval()
    
    correct, total, test_loss = 0, 0, 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
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




# Load trained FL model
def load_model():
    from cifar10_partition import CNN  # Import your model class
    model = CNN().to(device) # Initialize the model architecture
    model.load_state_dict(torch.load("global_model.pth", map_location=device))  # Load weights
    model.eval()
    return model



if __name__ == "__main__":
    print("🔵 Loading test data...")
    test_loader = load_test_data()
    
    print("🔵 Loading trained global model...")
    model = load_model()
    
    print("🔵 Evaluating model on test set...")
    accuracy, avg_loss, conf_matrix = evaluate_model(model, test_loader, device)
    
    print("🔵 Generating confusion matrix...")
    plot_confusion_matrix(conf_matrix)  
    
    print("✅ Model evaluation completed!")

    
    
    # Evaluate the model
    accuracy, avg_loss, conf_matrix = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {avg_loss:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix)
    


import os
if os.name == 'posix':
    os.dup2(os.open('/dev/null', os.O_WRONLY), 2)
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from collections import defaultdict
from FL_model_testing import evaluate_model, load_test_data

# Function to download and load CIFAR-10
def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    return trainset, testset

# Non-IID Partitioning using Dirichlet Distribution
def non_iid_partition(dataset, num_clients=10, alpha=7.0):
    num_classes = 10
    data_indices = np.array(range(len(dataset)))
    labels = np.array(dataset.targets)

    # Get class-wise indices
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    # Create Dirichlet distribution for each class
    client_dict = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        np.random.shuffle(class_indices[c])
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]

        # Split class data among clients
        split_indices = np.split(class_indices[c], proportions)
        for i, idx in enumerate(split_indices):
            client_dict[i].extend(idx)

    return client_dict

# Load CIFAR-10 dataset
trainset, testset = load_cifar10()

# Number of clients & partition skewness
num_clients = 10  # Adjust as needed
alpha = 0.7  # Lower alpha -> More imbalanced splits

# Get partitioned indices
client_data_indices = non_iid_partition(trainset, num_clients=num_clients, alpha=alpha)

# Convert partitioned indices into datasets for each client
client_datasets = {client: Subset(trainset, indices) for client, indices in client_data_indices.items()}

# Print dataset sizes per client
#for client, dataset in client_datasets.items():
#    print(f"Client {client}: {len(dataset)} samples")


import matplotlib
matplotlib.use("TkAgg")  # Force Matplotlib to use Tkinter backend
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

plt.ion()  # Turn on interactive mode


# Function to extract class distributions per client
def extract_class_distribution(client_data_indices, dataset):
    num_clients = len(client_data_indices)
    num_classes = 10  # CIFAR-10 has 10 classes
    class_counts = np.zeros((num_clients, num_classes), dtype=int)

    labels = np.array(dataset.targets)

    for client, indices in client_data_indices.items():
        for idx in indices:
            class_counts[client, labels[idx]] += 1

    return class_counts

# Extract class distributions per client
class_distributions = extract_class_distribution(client_data_indices, trainset)

# Define CIFAR-10 class labels
cifar10_labels = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Plot stacked bar chart for class distributions
fig, ax = plt.subplots(figsize=(12, 6))
width = 0.7  # Bar width
bottom = np.zeros(num_clients)  # Used for stacking bars

for class_idx in range(10):
    ax.bar(range(num_clients), class_distributions[:, class_idx], width,
           label=cifar10_labels[class_idx], bottom=bottom)
    bottom += class_distributions[:, class_idx]

ax.set_xlabel("Client ID")
ax.set_ylabel("Number of samples")
ax.set_title("Non-IID class distribution across clients")
ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1))
plt.xticks(range(num_clients), [f"Client {i}" for i in range(num_clients)])
plt.tight_layout()

# Show the plot
plt.pause(0.1)  # Allow time for graph to update
plt.draw()



import torch.nn as nn
import torch.optim as optim
import random

# Define a simple CNN model for CIFAR-10
class CNN(nn.Module):
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

# Function to train a model on a client's local dataset
def train_client(model, dataloader, epochs=5, lr=0.005):
    model.train()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)  # Helps prevent model collapse
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.7, weight_decay=1e-4)  #  Adds weight decay

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients before update
            optimizer.step()
            total_loss += loss.item()
            # Debug print for first and last batch of each epoch
            if batch_idx == 0 or batch_idx == len(dataloader) - 1:
                print(f"Client Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(dataloader):.4f}")


def federated_averaging(global_model, client_models, client_data_sizes):
    """
    Federated averaging with sample-size weighting.

    global_model: The central model being updated.
    client_models: List of models from clients.
    client_data_sizes: List of sample sizes for each client.
    """
    global_dict = global_model.state_dict()
    
    # Compute total number of samples across selected clients
    total_samples = sum(client_data_sizes)
    
    # Initialize an empty dictionary for weighted sum
    weighted_avg = {key: torch.zeros_like(value) for key, value in global_dict.items()}
    # Perform weighted averaging
    for client_idx, client_model in enumerate(client_models):  
        client_weight = client_data_sizes[client_idx] / total_samples  
        client_dict = client_model.state_dict()

        for key in weighted_avg.keys():
        # Add weighted contribution of client's weights
            weighted_avg[key] = weighted_avg[key].to(torch.float32)  # Ensure float before updating
            weighted_avg[key] += (client_weight * client_dict[key].to(torch.float32))


        print(f"Aggregated Weights for {key}: {weighted_avg[key].sum()}")


    # Load the new weighted averaged model
    global_model.load_state_dict(weighted_avg)

    # Save the global model
    torch.save(global_model.state_dict(), "global_model.pth")
    print("Global model saved as global_model.pth")

    return global_model

    
# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize global model
global_model = CNN().to(device)

# Convert partitioned datasets into DataLoaders
batch_size = 64
client_loaders = {
    client: DataLoader(client_datasets[client], batch_size=batch_size, shuffle=True, num_workers=0)
    for client in client_datasets
}

test_loader = load_test_data()

# Perform Federated Learning (5 rounds)
num_rounds = 5
num_clients_per_round = 7  # Select subset of clients each round



for round_num in range(num_rounds):
    print(f"\n=== FL Round {round_num + 1} ===")

    # Select random subset of clients (keeping your logic)
    #selected_clients = sorted(client_loaders.keys(), key=lambda x: len(set(client_datasets[x].dataset.targets)), reverse=True)[:num_clients_per_round]
    # Choose clients randomly instead of by class distribution
    selected_clients = random.sample(list(client_loaders.keys()), num_clients_per_round)

    # Train models locally on selected clients
    client_models = []
    for client in selected_clients:
        print(f"Training Client {client}...")
        local_model = CNN().to(device)
        local_model.load_state_dict(global_model.state_dict())
        train_client(local_model, client_loaders[client], epochs=4)  # local epoch per round
        client_models.append(local_model)
        # Evaluate client model on its local data (new debugging step)
        client_accuracy, _, _ = evaluate_model(local_model, client_loaders[client], device)
        print(f"Client {client} Accuracy on Local Data: {client_accuracy:.4f}")

    # Aggregate models using FedAvg
    client_data_sizes = [len(client_loaders[client].dataset) for client in selected_clients]
    global_model = federated_averaging(global_model, client_models, client_data_sizes)  # Assign the result to global_model

    # Evaluate global model on test set after this round (new debugging step)
    accuracy, _, conf_matrix = evaluate_model(global_model, test_loader, device)
    print(f"Round {round_num + 1} - Global Model Test Accuracy: {accuracy:.4f}")

print("\nFederated Learning Training Complete!")



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.backends.mkl as mkl


# Define the neural network
class SnakeNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SnakeNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Custom dataset
class SnakeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Load and preprocess data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        features = data[['snake_head_x', 'snake_head_y', 'food_x', 'food_y', 'snake_length']].values
        labels = data['direction'].values

        # Normalize features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        return features, labels
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None, None
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_path} is empty.")
        return None, None
    except KeyError as e:
        print(f"Error: The column {e} is missing from the CSV file.")
        return None, None

# Train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Accuracy: {100 * correct / total:.2f}%')

def main():
    # Hyperparameters
    input_size = 5
    hidden_size = 128  # Increased from 64
    output_size = 4
    batch_size = 64  # Increased from 32
    learning_rate = 0.01  # Increased from 0.001
    num_epochs = 100  # Increased from 50

    # Set the number of threads for OpenMP-based parallelism
    torch.set_num_threads(torch.get_num_threads())

    # Use the fastest available algorithm for convolutions
    torch.backends.cudnn.benchmark = True

    # Use mixed precision training if available
    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Initialize the model, loss function, and optimizer
    model = SnakeNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load data
    features, labels = load_data('snake_training_data.csv')
    if features is None or labels is None:
        return

    # Check if we have enough data
    if len(features) < 1000:
        print("Warning: The dataset is smaller than expected. Consider generating more data.")

    # Print feature statistics
    print("Feature means:", np.mean(features, axis=0))
    print("Feature std devs:", np.std(features, axis=0))

    # Check class balance
    unique, counts = np.unique(labels, return_counts=True)
    print("Class distribution:", dict(zip(unique, counts)))

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Create datasets and dataloaders
    train_dataset = SnakeDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = SnakeDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Train the model with mixed precision
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, scaler)

    # Save the model in PyTorch format
    torch.save(model.state_dict(), 'snake_model.pth')
    print("Model saved in PyTorch format as 'snake_model.pth'")

    # Save the model in ONNX format
    dummy_input = torch.randn(1, input_size)
    torch.onnx.export(model, dummy_input, "snake_model.onnx", verbose=True)
    print("Model saved in ONNX format as 'snake_model.onnx'")

if __name__ == "__main__":
    main()
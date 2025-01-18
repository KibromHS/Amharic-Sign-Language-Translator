import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from model import ASLModel

# Initialize MediaPipe Hands
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

class ASLDataset(Dataset):
    """Custom dataset to load and preprocess sign language data."""

    def __init__(self, dataset_path):
        self.data = []
        self.labels = []
        self.label_map = {}  # Maps folder names to numeric labels
        self.load_data(dataset_path)

    def load_data(self, dataset_path):
        # Get folder names and sort them alphabetically
        folder_names = sorted(os.listdir(dataset_path))
        self.label_map = {name: idx for idx, name in enumerate(folder_names)}

        # Iterate through folders to load images and labels
        for folder_name, label in self.label_map.items():
            folder_path = os.path.join(dataset_path, folder_name)
            for img_file in os.listdir(folder_path):
                if img_file.endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(folder_path, img_file)
                    landmarks = self.extract_landmarks(img_path)
                    if landmarks is not None:
                        self.data.append(landmarks)
                        self.labels.append(label)

        # Convert to PyTorch tensors
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def extract_landmarks(self, img_path):
        """Extract hand landmarks from an image."""
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            landmarks = [lm for point in hand_landmarks.landmark for lm in (point.x, point.y, point.z)]
            return (np.array(landmarks, dtype=np.float32) - np.mean(landmarks)) / (np.std(landmarks) + 1e-6)
        return None  # Skip images without detectable hands

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(dataset_path, num_epochs=20, batch_size=32, learning_rate=0.001, save_path="model.pth"):
    """Train the ASL model and save it."""
    # Load dataset
    asl_dataset = ASLDataset(dataset_path)
    train_loader = DataLoader(asl_dataset, batch_size=batch_size, shuffle=True)

    # Define the model, loss, and optimizer
    model = ASLModel(num_classes=len(asl_dataset.label_map))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model training completed and saved as '{save_path}'.")

if __name__ == "__main__":
    dataset_path = "dataset"  # Path to dataset folders
    num_epochs = 20
    batch_size = 32
    learning_rate = 0.001
    save_path = "model.pth"

    # Train the model
    train_model(dataset_path, num_epochs, batch_size, learning_rate, save_path)

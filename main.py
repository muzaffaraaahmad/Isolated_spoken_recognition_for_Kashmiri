from data_preprocessing import extract_features
from train import train_model
from evaluate import evaluate_model
import os

# Define dataset path
dataset_path = "/home/muzaffar/Desktop/Research/papers/mono-model-word-spoken/paper-001-final/1-20 data/ORIGINAL-SLOW-1-20"

# Extract features and labels
features, labels = extract_features(dataset_path)

# Train model
model = train_model(features, labels)

# Evaluate model
evaluate_model(model, features, labels, unique_labels=np.unique(labels))

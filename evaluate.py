import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_val, y_val, unique_labels):
    y_true = np.argmax(y_val, axis=1)
    y_pred = np.argmax(model.predict(X_val), axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

    class_report = classification_report(y_true, y_pred, target_names=unique_labels, digits=4)
    print("Classification Report:\n", class_report)

    conf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(conf_matrix, unique_labels)

def plot_confusion_matrix(conf_matrix, class_labels):
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='viridis', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Normalized)')
    plt.show()

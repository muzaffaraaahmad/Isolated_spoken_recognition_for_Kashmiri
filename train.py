import numpy as np
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from model import create_blstm_model

def train_model(features, labels, num_folds=5, epochs=30, batch_size=32):
    unique_labels = np.unique(labels)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_map[label] for label in labels])
    categorical_labels = to_categorical(encoded_labels)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    fold = 1
    for train_idx, val_idx in kf.split(features):
        print(f"Training fold {fold}...")
        fold += 1

        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = categorical_labels[train_idx], categorical_labels[val_idx]

        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = categorical_labels.shape[1]

        model = create_blstm_model(input_shape, num_classes)

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)
    
    return model

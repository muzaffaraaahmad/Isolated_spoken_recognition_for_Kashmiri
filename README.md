# Isolated_spoken_recognition_for_Kashmiri

This repository contains a Bidirectional Long Short-Term Memory (BLSTM) model for speech recognition. The system extracts mel-spectrogram features from speech audio files, processes them, trains a deep learning model using K-fold cross-validation, and evaluates performance using classification metrics such as accuracy, confusion matrix, and classification report.



Project Structure:    
├── data_preprocessing.py  # Handles feature extraction and normalization  
├── model.py               # Defines the BLSTM model architecture  
├── train.py               # Handles model training using K-fold cross-validation  
├── evaluate.py            # Computes evaluation metrics (accuracy, confusion matrix)  
├── utils.py               # Contains utility functions  
├── main.py                # Main script to run feature extraction, training, and evaluation  
├── README.md              # Project documentation  


Installation:    

1. Clone the repository    
git clone https://github.com/your-repo/blstm-speech-recognition.git    



2. Install dependencies    
pip install -r requirements.txt    



Dataset format:    
dataset/    
│── class_1/    
│   ├── file1.wav    
│   ├── file2.wav    
│── class_2/    
│   ├── file3.wav    
│   ├── file4.wav    
...    
    


How to Run the Project    
Step 1: Extract Features    
Modify main.py to include the correct dataset path. Then run:    

python main.py    



Step 2: Train the Model    
Training is handled using 5-fold cross-validation. You can modify the number of epochs and batch size in train.py.    
    
Step 3: Evaluate the Model    
After training, the model computes classification metrics and plots a confusion matrix.    



Model Architecture    
Input: Mel-spectrogram features extracted from .wav files    
Hidden Layers:    
    Bidirectional LSTM (64 units) → Dropout    
    Bidirectional LSTM (64 units) → Dropout    
    Dense (64 units, ReLU) → Dropout    
Output Layer: Softmax activation for multi-class classification    


Performance Metrics:    
The model's performance is measured using:
✅ Accuracy        
✅ Precision, Recall, and F1-score    
✅ Confusion Matrix    





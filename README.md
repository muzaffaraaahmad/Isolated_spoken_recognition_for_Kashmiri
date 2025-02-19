# Isolated_spoken_recognition_for_Kashmiri

This repository contains a Bidirectional Long Short-Term Memory (BLSTM) model for speech recognition. The system extracts mel-spectrogram features from speech audio files, processes them, trains a deep learning model using K-fold cross-validation, and evaluates performance using classification metrics such as accuracy, confusion matrix, and classification report.





**Project Structure:**     
├── data_preprocessing.py  # Handles feature extraction and normalization  
├── model.py               # Defines the BLSTM model architecture  
├── train.py               # Handles model training using K-fold cross-validation  
├── evaluate.py            # Computes evaluation metrics (accuracy, confusion matrix)  
├── utils.py               # Contains utility functions  
├── main.py                # Main script to run feature extraction, training, and evaluation  
├── README.md              # Project documentation  





**Installation:**    

*1. Clone the repository*     
git clone https://github.com/your-repo/blstm-speech-recognition.git    



*2. Install dependencies*    
pip install -r requirements.txt    






**Dataset format:** 

dataset/    
│── class_1/    
│   ├── file1.wav    
│   ├── file2.wav    
│── class_2/    
│   ├── file3.wav    
│   ├── file4.wav    
...    
    





**How to Run the Project:**        

*Step 1: Extract Features*    
Modify main.py to include the correct dataset path. Then run:    

python main.py    


*Step 2: Train the Model*    
Training is handled using 5-fold cross-validation. You can modify the number of epochs and batch size in train.py. 


    
*Step 3: Evaluate the Model*    
After training, the model computes classification metrics and plots a confusion matrix.    





**Model Architecture:**     
Input: Mel-spectrogram features extracted from .wav files    
Hidden Layers:    
    Bidirectional LSTM (64 units) → Dropout    
    Bidirectional LSTM (64 units) → Dropout    
    Dense (64 units, ReLU) → Dropout    
Output Layer: Softmax activation for multi-class classification    


**Performance Metrics:**    
The model's performance is measured using:
✅ Accuracy        
✅ Precision, Recall, and F1-score    
✅ Confusion Matrix 



**Results:** 


# Testing Accuracy, Recall, Precision, and F1-score of the Proposed Bi-directional LSTM Model

| **Class**  | **Precision** | **Recall** | **F1-score** | **Testing Accuracy** |
|------------|-------------|------------|-------------|------------------|
| Akh       | 85.71%      | 83.33%     | 84.51%      | 77.78%          |
| Zi        | 91.12%      | 88.33%     | 89.70%      | 83.33%          |
| Thre      | 90.91%      | 88.89%     | 89.89%      | 88.89%          |
| Tsor      | 91.29%      | 84.44%     | 87.73%      | 86.94%          |
| Paanch    | 86.46%      | 83.33%     | 84.87%      | 79.17%          |
| Sheh      | 82.72%      | 87.78%     | 85.18%      | 84.17%          |
| Sath      | 84.14%      | 86.94%     | 85.52%      | 86.39%          |
| Aeth      | 84.34%      | 77.78%     | 80.92%      | 89.44%          |
| Nav       | 88.83%      | 86.11%     | 87.45%      | 86.11%          |
| Dah       | 77.10%      | 84.17%     | 80.48%      | 83.33%          |
| Kah       | 79.95%      | 86.39%     | 83.04%      | 85.00%          |
| Bah       | 83.69%      | 86.94%     | 85.29%      | 82.78%          |
| Truwah    | 87.85%      | 86.39%     | 87.11%      | 86.94%          |
| Czhodah   | 84.57%      | 79.17%     | 81.78%      | 87.78%          |
| Pandah    | 85.47%      | 85.00%     | 85.24%      | 85.56%          |
| Shurah    | 88.51%      | 85.56%     | 87.01%      | 88.89%          |
| Sadah     | 80.32%      | 82.78%     | 81.53%      | 86.39%          |
| Ardah     | 84.43%      | 88.89%     | 86.60%      | 84.44%          |
| Kunvuh    | 83.20%      | 89.44%     | 86.21%      | 83.89%          |
| Vuh       | 87.79%      | 83.89%     | 85.80%      | 88.33%          |



## Publication ##    

@article{Dar2025,
  title = {Bi-directional LSTM-based isolated spoken word recognition for Kashmiri language utilizing Mel-spectrogram feature},
  volume = {231},
  ISSN = {0003-682X},
  url = {http://dx.doi.org/10.1016/j.apacoust.2024.110505},
  DOI = {10.1016/j.apacoust.2024.110505},
  journal = {Applied Acoustics},
  publisher = {Elsevier BV},
  author = {Dar,  Muzaffar Ahmad and Pushparaj,  Jagalingam},
  year = {2025},
  month = mar,
  pages = {110505}
}







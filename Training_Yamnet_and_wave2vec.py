import os
import numpy as np
import librosa
import torch
import torchaudio
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# Paths
DATASET_DIR = "processed_data"  # Folder with preprocessed .wav files
CLASS_LABELS = {"crying": 0, "screaming": 1, "normal": 2}

# Load and preprocess dataset
def load_audio_files(dataset_dir):
    file_paths = []
    labels = []

    for file_name in os.listdir(dataset_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(dataset_dir, file_name)
            label = file_name.split("_")[0]  # Extract label from filename
            if label in CLASS_LABELS:
                file_paths.append(file_path)
                labels.append(CLASS_LABELS[label])

    return file_paths, np.array(labels)

# Load dataset
file_paths, labels = load_audio_files(DATASET_DIR)

# Split data (70% train, 15% val, 15% test)
train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.3, stratify=labels, random_state=42)
val_files, test_files, val_labels, test_labels = train_test_split(test_files, test_labels, test_size=0.5, stratify=test_labels, random_state=42)

# Data loader
def load_audio(file_path, sr=16000):
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    return y


yamnet_model = tf.keras.applications.MobileNetV2(input_shape=(96, 64, 1), include_top=False, weights=None)  # Replace with YAMNet
x = GlobalAveragePooling2D()(yamnet_model.output)
x = Dense(3, activation="softmax")(x)
yamnet_model = Model(inputs=yamnet_model.input, outputs=x)
yamnet_model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Convert dataset for YAMNet
def preprocess_yamnet(file_paths, labels):
    X = np.array([librosa.feature.melspectrogram(y=load_audio(f), sr=16000, n_mels=96) for f in tqdm(file_paths)])
    X = X[..., np.newaxis]  # Add channel dimension
    return X, np.array(labels)

X_train, y_train = preprocess_yamnet(train_files, train_labels)
X_val, y_val = preprocess_yamnet(val_files, val_labels)
X_test, y_test = preprocess_yamnet(test_files, test_labels)

# Train YAMNet
yamnet_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=3)

# Prepare dataset for Wav2Vec2
def preprocess_wav2vec2(file_paths, labels):
    inputs = processor([load_audio(f) for f in tqdm(file_paths)], sampling_rate=16000, return_tensors="pt", padding=True, truncation=True)
    return inputs.input_values, torch.tensor(labels)

X_train_w2v, y_train_w2v = preprocess_wav2vec2(train_files, train_labels)
X_val_w2v, y_val_w2v = preprocess_wav2vec2(val_files, val_labels)
X_test_w2v, y_test_w2v = preprocess_wav2vec2(test_files, test_labels)

# Define Wav2Vec2 training
optimizer = torch.optim.AdamW(wav2vec_model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

def train_wav2vec2(model, X_train, y_train, X_val, y_val, epochs=5):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train).logits
        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Evaluate on validation set
        with torch.no_grad():
            model.eval()
            val_outputs = model(X_val).logits
            val_loss = loss_fn(val_outputs, y_val)

train_wav2vec2(wav2vec_model, X_train_w2v, y_train_w2v, X_val_w2v, y_val_w2v)
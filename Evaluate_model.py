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

def evaluate_model(model, X_test, y_test, model_type="YAMNet"):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(f"Classification Report for {model_type}:\n", classification_report(y_test, y_pred_classes))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_LABELS.keys(), yticklabels=CLASS_LABELS.keys())
    plt.title(f"Confusion Matrix - {model_type}")
    plt.show()

evaluate_model(yamnet_model, X_test, y_test, model_type="YAMNet")
evaluate_model(wav2vec_model, X_test_w2v, y_test_w2v, model_type="Wav2Vec2")


# ---------------------------------------
def ensemble_predictions(yamnet_preds, wav2vec_preds):
    (yamnet_preds + wav2vec_preds) / 2

def plot_roc_curve(y_test, y_probs, model_name):
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # Convert labels to one-hot
    plt.figure(figsize=(7, 7))

    for i, label in enumerate(["Crying", "Screaming", "Normal"]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")  # Random classifier line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.show()
    

yamnet_preds = yamnet_model.predict(X_test)
wav2vec_preds = wav2vec_model(X_test_w2v).logits.detach().numpy()
ensemble_preds = ensemble_predictions(yamnet_preds, wav2vec_preds)

# Evaluate Ensemble
evaluate_model(ensemble_preds, y_test, model_type="Ensemble Model")
plot_roc_curve(y_test, yamnet_preds, "YAMNet")
plot_roc_curve(y_test, wav2vec_preds, "Wav2Vec2")
plot_roc_curve(y_test, ensemble_preds, "Ensemble Model")
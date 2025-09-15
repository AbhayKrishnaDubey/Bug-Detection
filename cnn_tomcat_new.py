import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_excel("D:\Research Minor Project\Sourcecode23-02-2022\data\Tomcat\Tomcat.xlsx")

# Data Preparation -----------------------------------------------------------
print("Original class distribution:")
print(df['status'].value_counts())

# Drop irrelevant columns
df = df.drop(columns=["id", "bug_id", "report_time", "commit", "files", "Unnamed: 10"], errors="ignore")

# Convert categorical target
label_encoder = LabelEncoder()
df['status'] = label_encoder.fit_transform(df['status'])

# Handle rare classes (minimum 5 samples per class)
min_samples = 5
counts = Counter(df['status'])
valid_classes = [k for k, v in counts.items() if v >= min_samples]
df_filtered = df[df['status'].isin(valid_classes)]

print("\nFiltered class distribution:")
print(df_filtered['status'].value_counts())

# Handle missing values
df_filtered['summary'] = df_filtered['summary'].fillna('')
df_filtered['description'] = df_filtered['description'].fillna('')

# Feature Engineering -------------------------------------------------------
# Fit TF-IDF only on training data later
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1,2), stop_words='english')

# Prepare text data
text_data = df_filtered['summary'] + ' ' + df_filtered['description']

# Split data first to avoid data leakage
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    text_data, df_filtered['status'], 
    test_size=0.2, 
    random_state=42,
    stratify=df_filtered['status']
)

# Apply TF-IDF separately
X_train_text = tfidf.fit_transform(X_train_raw).toarray()
X_test_text = tfidf.transform(X_test_raw).toarray()

# Numerical features
scaler = StandardScaler()
numeric_features = df_filtered[['report_timestamp', 'commit_timestamp']].fillna(0)
X_train_num = scaler.fit_transform(numeric_features.loc[y_train_raw.index])
X_test_num = scaler.transform(numeric_features.loc[y_test_raw.index])

# Combine features
X_train = np.hstack((X_train_text, X_train_num))
X_test = np.hstack((X_test_text, X_test_num))

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train_raw)

# One-hot encode
y_train = keras.utils.to_categorical(y_train_res)
y_test = keras.utils.to_categorical(y_test_raw)

# Reshape for CNN
X_train = np.expand_dims(X_train_res, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Model Architecture --------------------------------------------------------
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', 
               input_shape=input_shape, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Flatten(),
        
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True),  # Reduced patience
    ReduceLROnPlateau(monitor='val_auc', factor=0.2, patience=3, min_lr=1e-6),  # Reduced patience
    ModelCheckpoint('best_model.h5', monitor='val_auc', mode='max', save_best_only=True)
]

# Build and train model (30 epochs)
model = build_model((X_train.shape[1], 1), y_train.shape[1])
history = model.fit(
    X_train, y_train,
    epochs=30,  # Reduced from 100 to 30
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# Enhanced Evaluation ------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Metrics
    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(X_test, y_test)
    
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"AUC: {test_auc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true_classes, 
        y_pred_classes, 
        target_names=label_encoder.classes_[valid_classes]
    ))
    
    # Confusion matrix
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_[valid_classes],
                yticklabels=label_encoder.classes_[valid_classes])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

evaluate_model(model, X_test, y_test)

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    metrics = ['loss', 'auc', 'precision', 'recall']
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i+1)
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(metric.upper())
        plt.xlabel('Epoch')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)
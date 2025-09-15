import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_excel("D:\Research Minor Project\Sourcecode23-02-2022\data\JDT\JDT.xlsx")

# Data Preparation -----------------------------------------------------------
print("Original class distribution:")
print(df['status'].value_counts())

# Drop irrelevant columns
df = df.drop(columns=["id", "bug_id", "report_time", "commit", "files", "Unnamed: 10"], errors="ignore")

# Convert categorical target
label_encoder = LabelEncoder()
df['status'] = label_encoder.fit_transform(df['status'])

# Handle rare classes
min_samples = 5
counts = Counter(df['status'])
valid_classes = [k for k, v in counts.items() if v >= min_samples]
df_filtered = df[df['status'].isin(valid_classes)]

print("\nFiltered class distribution:")
print(df_filtered['status'].value_counts())

# Handle missing values
df_filtered['summary'] = df_filtered['summary'].fillna('').str.lower()
df_filtered['description'] = df_filtered['description'].fillna('').str.lower()

# Feature Engineering -------------------------------------------------------
# Text processing
tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1,3), stop_words='english', sublinear_tf=True)
text_data = df_filtered['summary'] + ' [SEP] ' + df_filtered['description']

# Split data
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    text_data, df_filtered['status'], 
    test_size=0.2, 
    random_state=42,
    stratify=df_filtered['status']
)

# Apply TF-IDF
X_train_text = tfidf.fit_transform(X_train_raw).toarray()
X_test_text = tfidf.transform(X_test_raw).toarray()

# Numerical features
scaler = StandardScaler()
numeric_cols = ['report_timestamp', 'commit_timestamp']
numeric_features = df_filtered[numeric_cols].fillna(0)
X_train_num = scaler.fit_transform(numeric_features.loc[y_train_raw.index])
X_test_num = scaler.transform(numeric_features.loc[y_test_raw.index])

# Combine features
X_train = np.hstack((X_train_text, X_train_num))
X_test = np.hstack((X_test_text, X_test_num))

# Handle class imbalance
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train_raw)

# One-hot encode
y_train = keras.utils.to_categorical(y_train_res)
y_test = keras.utils.to_categorical(y_test_raw)

# Reshape for CNN
X_train = np.expand_dims(X_train_res, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Optimized Model Architecture with Faster Learning ---------------------------------
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(256, kernel_size=5, activation='relu', 
               input_shape=input_shape, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(384, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dropout(0.4),
        
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(num_classes, activation='softmax')
    ])
    
    # More aggressive optimizer settings
    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,  # Increased base learning rate
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False
    )
    
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.F1Score(name='f1_score')
        ]
    )
    return model

# Learning rate scheduler for dynamic adjustment
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr  # Keep high LR for first 10 epochs
    elif epoch < 20:
        return lr * 0.5  # Reduce after 10 epochs
    elif epoch < 30:
        return lr * 0.2  # Reduce further after 20 epochs
    else:
        return lr * 0.1  # Final reduction

# Enhanced Callbacks
callbacks = [
    EarlyStopping(monitor='val_f1_score', patience=10, mode='max', restore_best_weights=True),
    ReduceLROnPlateau(
        monitor='val_f1_score', 
        factor=0.5,  # More aggressive reduction
        patience=3,  # Less patience
        min_lr=1e-5,  # Higher minimum learning rate
        verbose=1
    ),
    ModelCheckpoint('best_model.h5', monitor='val_f1_score', mode='max', save_best_only=True),
    LearningRateScheduler(lr_scheduler, verbose=1)
]

# Build and train model
model = build_model((X_train.shape[1], 1), y_train.shape[1])
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,  # Adjusted batch size
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# Evaluation ------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    test_loss, test_acc, test_precision, test_recall, test_auc, test_f1 = model.evaluate(X_test, y_test)
    
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"AUC: {test_auc:.4f}")
    print(f"F1-Score: {np.mean(test_f1):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        y_true_classes, 
        y_pred_classes, 
        target_names=label_encoder.classes_[valid_classes]
    ))
    
    plt.figure(figsize=(10,8))
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=label_encoder.classes_[valid_classes],
                yticklabels=label_encoder.classes_[valid_classes])
    plt.title('Confusion Matrix (% of True Class)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

evaluate_model(model, X_test, y_test)

# Plot training history with learning rate
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

# Print final learning rate
final_lr = keras.backend.get_value(model.optimizer.lr)
print(f"\nFinal Learning Rate: {final_lr:.6f}")
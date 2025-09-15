import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, TimeDistributed, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Load and preprocess data
df = pd.read_excel("D:\Research Minor Project\Sourcecode23-02-2022\data\Tomcat\Tomcat.xlsx")
df = df.drop(columns=["id", "bug_id", "Unnamed: 10", "report_time"], errors="ignore")

# Encode target
if 'status' in df.columns and df['status'].dtype == 'object':
    label_encoder = LabelEncoder()
    df['status'] = label_encoder.fit_transform(df['status'])

# Encode categorical features
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Scale numerical features
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 2. Prepare data for HAN
X = df.drop(columns=['status'])
y = keras.utils.to_categorical(df['status'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create sequences (3 timesteps per sample)
def create_sequences(features, seq_length=3):
    return np.array([np.tile(row, (seq_length, 1)) for row in features.values])

X_train_seq = create_sequences(X_train)  # (samples, timesteps, features)
X_test_seq = create_sequences(X_test)

# 3. Build improved HAN Model with regularization, batch normalization, and enhanced architecture
def build_improved_han_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Word-level attention
    word_encoded = TimeDistributed(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))(inputs)
    word_encoded = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)))(word_encoded)
    word_encoded = BatchNormalization()(word_encoded)  # Adding Batch Normalization
    
    # Attention mechanism
    attention = TimeDistributed(Dense(1, activation='tanh'))(word_encoded)
    attention = tf.nn.softmax(attention, axis=1)
    word_context = tf.reduce_sum(word_encoded * attention, axis=1)
    
    # Document-level encoding
    doc_encoded = Bidirectional(LSTM(128, kernel_regularizer=l2(0.01)))(word_encoded)
    
    # Combine
    combined = Concatenate()([word_context, doc_encoded])
    combined = Dropout(0.5)(combined)  # Adding dropout to the combined layer
    
    # Classifier
    outputs = Dense(num_classes, activation='softmax')(Dense(256, activation='relu')(combined))
    
    return Model(inputs=inputs, outputs=outputs)

# 4. Use a different optimizer with a learning rate schedule
optimizer = keras.optimizers.Adam(learning_rate=0.0005)

# 5. Rebuild and compile the model
model = build_improved_han_model((X_train_seq.shape[1], X_train_seq.shape[2]), y_train.shape[1])
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 6. Train with early stopping, learning rate scheduler, and model checkpoint
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = model.fit(X_train_seq, y_train,
                   epochs=100,
                   batch_size=64,  # Increased batch size
                   validation_data=(X_test_seq, y_test),
                   callbacks=[EarlyStopping(patience=5), lr_scheduler])

# 7. Evaluate the model
test_loss, test_acc = model.evaluate(X_test_seq, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# 8. Plot the training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()
plt.show()

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("D:\Research Minor Project\Sourcecode23-02-2022\data\JDT\JDT.xlsx")

# Drop irrelevant columns
df = df.drop(columns=["id", "bug_id", "Unnamed: 10", "report_time"], errors="ignore")

# Convert categorical target column (Assuming 'status' is the target)
if 'status' in df.columns and df['status'].dtype == 'object':
    label_encoder = LabelEncoder()
    df['status'] = label_encoder.fit_transform(df['status'])

# Identify categorical columns (excluding target)
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Encode categorical features
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Scale numerical features
scaler = StandardScaler()
numeric_features = df.select_dtypes(include=['int64', 'float64'])
df[numeric_features.columns] = scaler.fit_transform(df[numeric_features.columns])

# Define feature matrix (X) and target variable (y)
X = df.drop(columns=['status'], errors='ignore')
y = df['status']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest Classifier with improved parameters
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,  # Limit depth to prevent overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Cross-validation for better accuracy estimation
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print(f"Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%")

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance Analysis
feature_importance = rf_model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10, 5))
plt.barh([feature_names[i] for i in sorted_idx[:10]], feature_importance[sorted_idx[:10]], color='teal')
plt.xlabel("Feature Importance")
plt.title("Top 10 Important Features")
plt.gca().invert_yaxis()
plt.show()

# Plot training vs validation accuracy and loss
train_accuracy = rf_model.score(X_train, y_train)
train_loss = 1 - train_accuracy
test_loss = 1 - test_accuracy

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(['Train Accuracy', 'Test Accuracy'], [train_accuracy, test_accuracy], color=['blue', 'orange'])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Training vs Test Accuracy')

plt.subplot(1, 2, 2)
plt.bar(['Train Loss', 'Test Loss'], [train_loss, test_loss], color=['blue', 'orange'])
plt.ylim(0, 1)
plt.ylabel('Loss')
plt.title('Training vs Test Loss')

plt.show()
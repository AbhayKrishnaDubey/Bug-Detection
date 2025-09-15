import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("D:\Research Minor Project\Sourcecode23-02-2022\data\Tomcat\Tomcat.xlsx")

# Drop irrelevant columns
df = df.drop(columns=["id", "bug_id", "Unnamed: 10", "report_time"], errors="ignore")

# Convert categorical target column (Assuming 'status' is the target)
if 'status' in df.columns and df['status'].dtype == 'object':
    label_encoder = LabelEncoder()
    df['status'] = label_encoder.fit_transform(df['status'])  # Convert labels to numeric

# Identify categorical columns (excluding target)
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Encode categorical features (if any)
if categorical_columns:  # Ensure the list isn't empty
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Define feature matrix (X) and target variable (y)
X = df.drop(columns=['status'], errors='ignore')  # Drop the target column
y = df['status']  # Use label-encoded target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost model
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y)),
    eval_metric=['mlogloss', 'merror'],
    use_label_encoder=False,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8
)

# Train the model
history = model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Extract training history
results = model.evals_result()

# Plot validation vs test accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(1 - np.array(results['validation_0']['merror']), label='Train Accuracy')
plt.plot(1 - np.array(results['validation_1']['merror']), label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(results['validation_0']['mlogloss'], label='Train Loss')
plt.plot(results['validation_1']['mlogloss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()


# Show the plots
plt.show()

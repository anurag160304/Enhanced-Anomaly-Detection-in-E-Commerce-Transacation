
# Enhanced Anomaly Detection Model with Additional Improvements

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from keras.models import Sequential
from keras.layers import Dense

# Seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset (E-commerce transaction data with additional features)
normal_data = np.random.normal(loc=50, scale=10, size=(1000, 3))  # Normal transactions
anomalous_data = np.random.uniform(low=100, high=150, size=(50, 3))  # Anomalous transactions

# Add categorical feature (e.g., Transaction Category)
categories = np.random.choice(['Electronics', 'Fashion', 'Groceries'], 1050)
customer_ids = np.random.randint(1, 200, 1050)

data = np.vstack([normal_data, anomalous_data])
labels = np.hstack([np.zeros(1000), np.ones(50)])  # 0: Normal, 1: Anomaly

# Create DataFrame
columns = ['Transaction_Amount', 'Frequency', 'Time_Delta']
df = pd.DataFrame(data, columns=columns)
df['Label'] = labels
df['Category'] = categories
df['Customer_ID'] = customer_ids

# Add time-based and statistical features
df['Hour_of_Day'] = np.random.randint(0, 24, df.shape[0])
df['Day_of_Week'] = np.random.randint(1, 8, df.shape[0])
df['Transaction_Mean'] = df['Transaction_Amount'].rolling(window=10).mean().fillna(0)
df['Transaction_Std'] = df['Transaction_Amount'].rolling(window=10).std().fillna(0)

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['Transaction_Amount', 'Frequency', 'Time_Delta', 'Transaction_Mean', 'Transaction_Std']])

# Isolation Forest
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
df['IsolationForest_Score'] = isolation_forest.fit_predict(scaled_features)
df['IsolationForest_Anomaly'] = df['IsolationForest_Score'].apply(lambda x: 1 if x == -1 else 0)

# Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
df['LOF_Anomaly'] = lof.fit_predict(scaled_features)
df['LOF_Anomaly'] = df['LOF_Anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Autoencoder
autoencoder = Sequential([
    Dense(16, activation='relu', input_dim=scaled_features.shape[1]),
    Dense(8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(scaled_features.shape[1], activation='linear')
])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(scaled_features, scaled_features, epochs=50, batch_size=32, verbose=1)

# Autoencoder Anomaly Detection
reconstructions = autoencoder.predict(scaled_features)
mse = np.mean(np.square(scaled_features - reconstructions), axis=1)
threshold = np.percentile(mse, 95)
df['Autoencoder_Anomaly'] = mse > threshold

# Evaluation Metrics
print("Classification Report (Isolation Forest):")
print(classification_report(df['Label'], df['IsolationForest_Anomaly'], target_names=["Normal", "Anomaly"]))

print("Classification Report (LOF):")
print(classification_report(df['Label'], df['LOF_Anomaly'], target_names=["Normal", "Anomaly"]))

roc_auc = roc_auc_score(df['Label'], df['Autoencoder_Anomaly'])
print(f"ROC-AUC Score (Autoencoder): {roc_auc}")

precision, recall, _ = precision_recall_curve(df['Label'], mse)
pr_auc = auc(recall, precision)
print(f"Precision-Recall AUC (Autoencoder): {pr_auc}")

# Visualizations
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlations")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(df['Transaction_Amount'], df['Frequency'], c=df['IsolationForest_Anomaly'], cmap='coolwarm', label='Predicted (IF)')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.title('Isolation Forest: Anomalies vs Normal Transactions')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Autoencoder)')
plt.legend()
plt.grid(True)
plt.show()

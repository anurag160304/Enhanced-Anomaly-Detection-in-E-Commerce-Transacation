# Enhanced-Anomaly-Detection-in-E-Commerce-Transacation
# Enhanced Anomaly Detection Model

This project implements an enhanced anomaly detection system leveraging multiple machine learning approaches. The model is designed to detect anomalous transactions in an e-commerce dataset, integrating statistical features, categorical data, and time-based characteristics for improved accuracy.

## Features

- **Synthetic Dataset Generation**:
  - Simulates normal and anomalous e-commerce transactions.
  - Includes categorical and temporal features for a realistic dataset.

- **Preprocessing**:
  - Feature scaling using `StandardScaler`.
  - Statistical feature engineering (e.g., rolling mean, standard deviation).

- **Anomaly Detection Techniques**:
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - Autoencoder-based anomaly detection with reconstruction loss.

- **Evaluation Metrics**:
  - Classification reports.
  - ROC-AUC and Precision-Recall AUC for autoencoders.

- **Visualizations**:
  - Heatmaps of feature correlations.
  - Scatter plots of transaction data with anomaly predictions.
  - Precision-Recall curve for Autoencoder.

## Dependencies

Ensure the following libraries are installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `keras`
- `tensorflow`

You can install these dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn keras tensorflow

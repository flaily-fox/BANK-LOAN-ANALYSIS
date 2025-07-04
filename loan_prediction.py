import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load Data ---
print("--- 1. Loading Data ---")
try:
    df = pd.read_csv('customer_loan_details.csv')
    print("Data loaded successfully.")
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
except FileNotFoundError:
    print("Error: 'customer_loan_details.csv' not found. Please make sure the file is in the correct directory.")
    exit()
    
# --- 2. Data Preprocessing ---
print("\n--- 2. Data Preprocessing ---")

# Handle Missing Values (if any)
print("\nChecking for missing values:")
print(df.isnull().sum())
# For simplicity, we'll fill numerical NaNs with the mean and categorical with the mode.
# Based on the provided sample data, there might not be explicit NaNs, but it's good practice.
for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].mean(), inplace=True)
print("\nMissing values after handling:")
print(df.isnull().sum())

# Identify Categorical and Numerical Features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")

# Remove 'applicantId' as it's not a feature for predictions
if 'applicantId' in df.columns:
    df = df.drop('applicantId', axis=1)
    if 'applicantId' in categorical_features:
        categorical_features.remove('applicantId')
    print("\n'applicantId' column removed.")

# Drop 'loan_decision_type' from categorical features as it's our target variable
if 'loan_decision_type' in categorical_features:
    categorical_features.remove('loan_decision_type')

# Label Encoding for Categorical Features
print("\nApplying Label Encoding for categorical features...")
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    label_encoders[feature] = le
    print(f"  - Encoded '{feature}'. Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Encode the target variable 'loan_decision_type'
print("\nEncoding target variable 'loan_decision_type'...")
le_loan_decision = LabelEncoder()
df['loan_decision_type'] = le_loan_decision.fit_transform(df['loan_decision_type'])
print(f"  - Encoded 'loan_decision_type'. Mapping: {dict(zip(le_loan_decision.classes_, le_loan_decision.transform(le_loan_decision.classes_)))}")

# Separate features (X) and target (y)
X = df.drop('loan_decision_type', axis=1)
y = df['loan_decision_type']

print(f"\nShape of X (features): {X.shape}")
print(f"Shape of y (target): {y.shape}")
print("\nFirst 5 rows of X after encoding:")
print(X.head())

# Normalization (Standard Scaling) for Numerical Features
# This is crucial for PCA as it's sensitive to feature scales.
print("\nApplying Standard Scaling for numerical features...")
scaler = StandardScaler()
# Ensure only numerical columns are scaled, excluding those already encoded as integers
# The categorical features are now integers, but StandardScaler would treat them as numerical.
# We need to scale only the original numerical features like 'age', 'credit_score', 'income', 'debts'.
# Let's re-identify numerical features that need scaling *after* label encoding and dropping applicantId.
numerical_cols_to_scale = ['age', 'credit_score', 'income', 'debts']
X[numerical_cols_to_scale] = scaler.fit_transform(X[numerical_cols_to_scale])
print("Standard Scaling applied.")
print("\nFirst 5 rows of X after scaling:")
print(X.head())

# PCA to Reduce Dimensions
print("\nApplying PCA for dimensionality reduction...")
# We'll choose the number of components such that 95% of variance is explained.
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X)

print(f"Original number of features: {X.shape[1]}")
print(f"Number of features after PCA (components explaining 95% variance): {X_pca.shape[1]}")
print(f"Explained variance ratio by each component: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {np.sum(pca.explained_variance_ratio_)}")

# --- 3. Model Building ---
print("\n--- 3. Model Building ---")

# Train/Test Split (70-30 ratio)
print("\nSplitting data into training and testing sets (70-30 ratio)...")
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42, stratify=y)
# stratify=y ensures that the proportion of target variable categories is the same in both train and test sets.

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Naive Bayes Classifier (GaussianNB as features are continuous after PCA)
print("\nInitializing and training Gaussian Naive Bayes classifier...")
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("Model training complete.")

# Cross-Validation for Performance Tuning
print("\nPerforming 5-fold Cross-Validation...")
cv_scores = cross_val_score(gnb, X_pca, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {np.mean(cv_scores):.4f}")
print(f"Standard deviation of cross-validation accuracy: {np.std(cv_scores):.4f}")

# --- 4. Evaluation ---
print("\n--- 4. Evaluation ---")

# Predictions on the test set
y_pred = gnb.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score on Test Set: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_loan_decision.classes_,
            yticklabels=le_loan_decision.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Precision, Recall, F1 Score
class_report = classification_report(y_test, y_pred, target_names=le_loan_decision.classes_)
print("\nClassification Report (Precision, Recall, F1 Score):")
print(class_report)

print("\n--- Project Complete ---")
print("The model has been trained and evaluated. The metrics above indicate its performance.")
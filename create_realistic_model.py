import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Create a realistic mock model that matches the test_data.csv structure
np.random.seed(42)

# Read the test data to get the correct feature structure
test_data = pd.read_csv('test_data.csv')
feature_names = [col for col in test_data.columns if col != 'target']
n_features = len(feature_names)

print(f"Creating model with {n_features} features: {feature_names[:5]}...")

# Generate training data that matches the test data structure
n_train_samples = 1000
X_train = np.random.randn(n_train_samples, n_features)

# Create realistic coefficients for churn prediction
coefficients = np.random.randn(n_features) * 0.5
# Make some features more important (like contract length, tenure, etc.)
coefficients[0] *= 2  # age
coefficients[1] *= -3  # tenure_months (longer tenure = less churn)
coefficients[2] *= 1.5  # monthly_charges

# Create target with some logic
y_train = (X_train @ coefficients + np.random.randn(n_train_samples) * 0.5 > 0).astype(int)

# Train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model created and saved as model.pkl")
print(f"Model accuracy on training data: {model.score(X_train, y_train):.3f}")

# Test with actual test data
X_test = test_data.drop('target', axis=1).values
y_test = test_data['target'].values
test_accuracy = model.score(X_test, y_test)
print(f"Model accuracy on test data: {test_accuracy:.3f}")

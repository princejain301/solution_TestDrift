import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic test data for customer churn prediction
n_samples = 200

# Create synthetic features
data = {
    'age': np.random.randint(18, 80, n_samples),
    'tenure_months': np.random.randint(1, 120, n_samples),
    'monthly_charges': np.random.uniform(20, 120, n_samples),
    'total_charges': np.random.uniform(100, 8000, n_samples),
    'contract_length': np.random.choice(['month-to-month', 'one_year', 'two_year'], n_samples),
    'payment_method': np.random.choice(['credit_card', 'bank_transfer', 'electronic_check', 'mailed_check'], n_samples),
    'internet_service': np.random.choice(['dsl', 'fiber_optic', 'no'], n_samples),
    'tech_support': np.random.choice(['yes', 'no'], n_samples),
    'online_security': np.random.choice(['yes', 'no'], n_samples),
    'paperless_billing': np.random.choice(['yes', 'no'], n_samples)
}

# Create target variable (churn) with some logical correlations
churn_probability = (
    0.1 +  # base probability
    0.3 * (data['contract_length'] == 'month-to-month').astype(int) +
    0.2 * (data['monthly_charges'] > 80).astype(int) +
    0.15 * (data['tenure_months'] < 12).astype(int) +
    0.1 * (data['tech_support'] == 'no').astype(int)
)

# Add some randomness
churn_probability = np.clip(churn_probability + np.random.normal(0, 0.1, n_samples), 0, 1)
data['target'] = (np.random.random(n_samples) < churn_probability).astype(int)

# Convert to DataFrame
df = pd.DataFrame(data)

# One-hot encode categorical variables for model compatibility
df_encoded = pd.get_dummies(df, columns=['contract_length', 'payment_method', 'internet_service', 'tech_support', 'online_security', 'paperless_billing'])

# Save the test data
df_encoded.to_csv('test_data.csv', index=False)

print(f"Generated test_data.csv with {n_samples} samples")
print(f"Features: {list(df_encoded.columns)}")
print(f"Churn rate: {df_encoded['target'].mean():.2%}")
print(f"Shape: {df_encoded.shape}")

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

# Create a simple mock model for testing purposes
np.random.seed(42)
X_dummy = np.random.rand(100, 4)
y_dummy = np.random.randint(0, 2, 100)

model = LogisticRegression()
model.fit(X_dummy, y_dummy)

# Save the model
with open('sample_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Sample model created and saved as sample_model.pkl")

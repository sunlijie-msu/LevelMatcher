import numpy as np
from xgboost import XGBRegressor

training_data_points = [
    (0.0, 0, 1.00), (0.5, 0, 0.99), (1.0, 0, 0.98), (1.5, 0, 0.95),
    (1.8, 0, 0.90), (2.0, 0, 0.80), (2.2, 0, 0.70), (2.5, 0, 0.60),
    (2.8, 0, 0.55), (3.0, 0, 0.50), (3.2, 0, 0.40), (3.5, 0, 0.20),
    (4.0, 0, 0.10), (5.0, 0, 0.01), (10.0, 0, 0.00), (20.0, 0, 0.00),
    (100.0, 0, 0.00),
    (0.0, 1, 0.00), (0.1, 1, 0.00), (0.2, 1, 0.00), (0.3, 1, 0.00),
    (0.4, 1, 0.00), (0.5, 1, 0.00), (0.6, 1, 0.00), (0.7, 1, 0.00),
    (0.8, 1, 0.00), (0.9, 1, 0.00), (1.0, 1, 0.00), (1.5, 1, 0.00),
    (2.0, 1, 0.00), (3.0, 1, 0.00), (5.0, 1, 0.00), (100.0, 1, 0.00)
]
X_train = np.array([[pt[0], pt[1]] for pt in training_data_points])
y_train = np.array([pt[2] for pt in training_data_points])

# Increased max_depth to 4
model = XGBRegressor(objective='binary:logistic', monotone_constraints='(-1, -1)', 
                     n_estimators=200, max_depth=4, random_state=42)
model.fit(X_train, y_train)

print(f"Prediction for Z=0.0, Veto=1: {model.predict([[0.0, 1]])[0]}")
print(f"Prediction for Z=0.0, Veto=0: {model.predict([[0.0, 0]])[0]}")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
X = np.linspace(0, 50, 100)
true_theta0 = 5
true_theta1 = 3
noise = np.random.randn(100) * 10
y = true_theta0 + true_theta1 * X + noise


st.title("🔍 Interactive Linear Regression Explorer")
st.markdown("Use the sliders to adjust **θ₀** (intercept) and **θ₁** (slope) and observe how the best fit line changes.")


theta0 = st.slider("θ₀ (Intercept)", min_value=-50.0, max_value=50.0, value=0.0, step=0.5)
theta1 = st.slider("θ₁ (Slope)", min_value=-5.0, max_value=10.0, value=0.0, step=0.1)

# Predicted line
y_pred = theta0 + theta1 * X

# Cost function (Mean Squared Error)
m = len(X)
cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)


st.write(f"### 📉 Cost (J): {cost:.2f}")

# Plotting
fig, ax = plt.subplots()
ax.scatter(X, y, label='Data (noisy)', color='skyblue', alpha=0.6)
ax.plot(X, y_pred, color='red', label=f'Prediction: y = {theta0:.1f} + {theta1:.1f}x')
ax.legend()
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.set_title("Best Fit Line Visualizer")

st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

# ---------------------------------------------------------
# Gradient Descent (Normalized)
# ---------------------------------------------------------
def gradient_descent_normalized(x, y, lr=0.01, n_iters=1000):
    n = len(x)
    b0, b1 = 0.0, 0.0
    b0_hist, b1_hist, cost_hist = [], [], []

    for i in range(n_iters):
        y_pred = b0 + b1 * x
        error = y - y_pred
        cost = (1/(2*n)) * np.sum(error**2)

        if np.isnan(cost) or cost > 1e8:
            break

        db0 = -(1/n) * np.sum(error)
        db1 = -(1/n) * np.sum(error * x)

        db0 = np.clip(db0, -1e5, 1e5)
        db1 = np.clip(db1, -1e5, 1e5)

        b0 -= lr * db0
        b1 -= lr * db1

        b0_hist.append(b0)
        b1_hist.append(b1)
        cost_hist.append(cost)

    return np.array(b0_hist), np.array(b1_hist), np.array(cost_hist)


# ---------------------------------------------------------
# 2D Contour Plot for GD Path
# ---------------------------------------------------------
def plot_cost_contour(b0_hist, b1_hist, x, y):
    b0_vals = np.linspace(min(b0_hist)-1, max(b0_hist)+1, 50)
    b1_vals = np.linspace(min(b1_hist)-1, max(b1_hist)+1, 50)

    B0, B1 = np.meshgrid(b0_vals, b1_vals)
    J = np.zeros_like(B0)

    for i in range(B0.shape[0]):
        for j in range(B0.shape[1]):
            y_pred = B0[i,j] + B1[i,j] * x
            J[i,j] = np.mean((y - y_pred)**2) / 2

    fig, ax = plt.subplots(figsize=(6,4))
    ax.contour(B0, B1, J, levels=30)
    ax.plot(b0_hist, b1_hist, 'r.-', label="GD Path")
    ax.set_xlabel("b₀")
    ax.set_ylabel("b₁")
    ax.set_title("Gradient Descent Path (Contour View)")
    ax.legend()
    return fig


# ---------------------------------------------------------
# 3D Plotly Cost Surface + GD Path
# ---------------------------------------------------------
def plot_3d_surface(b0_hist, b1_hist, x, y):
    b0_vals = np.linspace(min(b0_hist)-1, max(b0_hist)+1, 40)
    b1_vals = np.linspace(min(b1_hist)-1, max(b1_hist)+1, 40)

    B0, B1 = np.meshgrid(b0_vals, b1_vals)
    J = np.zeros_like(B0)

    for i in range(B0.shape[0]):
        for j in range(B0.shape[1]):
            y_pred = B0[i,j] + B1[i,j] * x
            J[i,j] = np.mean((y - y_pred) ** 2) / 2

    surface = go.Surface(z=J, x=B0, y=B1, colorscale="Viridis", opacity=0.8)

    gd_path = go.Scatter3d(
        x=b0_hist,
        y=b1_hist,
        z=[np.mean((y - (b0_hist[i] + b1_hist[i] * x))**2)/2 for i in range(len(b0_hist))],
        mode="lines+markers",
        marker=dict(size=4, color="red"),
        line=dict(color="red", width=3)
    )

    fig = go.Figure(data=[surface, gd_path])
    fig.update_layout(
        scene=dict(
            xaxis_title="b₀",
            yaxis_title="b₁",
            zaxis_title="Cost J"
        ),
        title="3D Cost Surface with Gradient Descent Path"
    )
    return fig


# ---------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------
def main():
    st.title("📊 Advanced Linear Regression with Train/Test + 3D GD")

    # -------------------------
    # TRAIN FILE
    # -------------------------
    st.header("1️⃣ Upload TRAIN File (YearsExperience + Salary)")
    train_file = st.file_uploader("Upload train Excel", type=["xlsx", "xls"])

    if train_file is None:
        return

    train_df = pd.read_excel(train_file)
    st.write("Train Data Preview:")
    st.dataframe(train_df.head())

    # -------------------------
    # TEST FILE
    # -------------------------
    st.header("2️⃣ Upload TEST File (YearsExperience + Salary)")
    test_file = st.file_uploader("Upload test Excel", type=["xlsx", "xls"])

    if test_file is None:
        return

    test_df = pd.read_excel(test_file)
    st.write("Test Data Preview:")
    st.dataframe(test_df.head())

    # -------------------------
    # Prepare data
    # -------------------------
    x_train_raw = train_df["YearsExperience"].astype(float).values
    y_train_raw = train_df["Salary"].astype(float).values

    x_test_raw = test_df["YearsExperience"].astype(float).values
    y_test_raw = test_df["Salary"].astype(float).values

    # Normalize
    x_mean, x_std = x_train_raw.mean(), x_train_raw.std()
    y_mean, y_std = y_train_raw.mean(), y_train_raw.std()

    x_train = (x_train_raw - x_mean) / x_std
    y_train = (y_train_raw - y_mean) / y_std

    # -------------------------
    # TRAIN MODEL
    # -------------------------
    st.header("3️⃣ Train Model (Gradient Descent)")
    lr = st.slider("Learning Rate", 0.0001, 1.0, 0.01)
    n_iters = st.slider("Iterations", 50, 5000, 500)

    if st.button("🚀 Train Model Now"):
        b0_hist, b1_hist, cost_hist = gradient_descent_normalized(x_train, y_train, lr, n_iters)

        st.session_state.update({
            "b0_hist": b0_hist,
            "b1_hist": b1_hist,
            "cost_hist": cost_hist,
            "x_train_raw": x_train_raw,
            "y_train_raw": y_train_raw,
            "x_test_raw": x_test_raw,
            "y_test_raw": y_test_raw,
            "x_mean": x_mean, "x_std": x_std,
            "y_mean": y_mean, "y_std": y_std
        })
        st.success("Model trained successfully!")

    if "b0_hist" not in st.session_state:
        return

    b0_hist = st.session_state["b0_hist"]
    b1_hist = st.session_state["b1_hist"]
    cost_hist = st.session_state["cost_hist"]

    # -------------------------
    # 3D COST SURFACE
    # -------------------------
    st.header("4️⃣ 3D Cost Surface with Gradient Descent Path")
    fig3d = plot_3d_surface(b0_hist, b1_hist, x_train, y_train)
    st.plotly_chart(fig3d, use_container_width=True)

    # -------------------------
    # Contour Plot (Top View)
    # -------------------------
    st.header("5️⃣ GD Contour Plot")
    fig2d = plot_cost_contour(b0_hist, b1_hist, x_train, y_train)
    st.pyplot(fig2d)

    # -------------------------
    # Regression line during GD
    # -------------------------
    st.header("6️⃣ Regression Line at GD Step")
    step = st.slider("Choose Step", 0, len(b0_hist)-1, len(b0_hist)-1)

    fig_gd, ax_gd = plt.subplots(figsize=(6,4))
    ax_gd.scatter(x_train_raw, y_train_raw, color='blue')

    x_line = np.linspace(min(x_train_raw), max(x_train_raw), 100)
    b0_step = b0_hist[step]
    b1_step = b1_hist[step]

    # Convert back to original scale
    b1_orig = (b1_step * y_std) / x_std
    b0_orig = y_mean + y_std * b0_step - b1_orig * x_mean

    y_line_step = b0_orig + b1_orig * x_line
    ax_gd.plot(x_line, y_line_step, 'g-', label=f"Iteration {step}")

    ax_gd.legend()
    st.pyplot(fig_gd)

    # -------------------------
    # FINAL REGRESSION LINE (Train + Test)
    # -------------------------
    st.header("7️⃣ Final Regression Line (Train + Test)")

    b0_final = b0_hist[-1]
    b1_final = b1_hist[-1]

    b1_f = (b1_final * y_std) / x_std
    b0_f = y_mean + y_std * b0_final - b1_f * x_mean

    fig_final, ax_final = plt.subplots(figsize=(6,4))
    ax_final.scatter(x_train_raw, y_train_raw, label="Train", color="blue")
    ax_final.scatter(x_test_raw, y_test_raw, label="Test", color="green")

    x_line = np.linspace(min(x_train_raw), max(x_train_raw), 100)
    ax_final.plot(x_line, b0_f + b1_f * x_line, 'r-', label="Final Model")

    ax_final.legend()
    st.pyplot(fig_final)

    # -------------------------
    # TEST METRICS
    # -------------------------
    st.header("8️⃣ Test Set Performance Metrics")

    y_test_pred = b0_f + b1_f * x_test_raw

    n = len(y_test_raw)
    r2 = r2_score(y_test_raw, y_test_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2)

    st.json({
        "MAE": mean_absolute_error(y_test_raw, y_test_pred),
        "MSE": mean_squared_error(y_test_raw, y_test_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test_raw, y_test_pred)),
        "R²": r2,
        "Adjusted R²": adj_r2
    })

# ---------------------------------------------------------
if __name__ == "__main__":
    main()

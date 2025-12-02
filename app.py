import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io
from PIL import Image

# --------------------------------------------------
# Gradient Descent Animation (GIF using Pillow)
# --------------------------------------------------
def animate_gd(x, y, b0_hist, b1_hist):
    frames = []
    x_line = np.linspace(np.min(x), np.max(x), 100)

    for i in range(len(b0_hist)):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x, y, color='blue', label="Data Points")

        y_line = b0_hist[i] + b1_hist[i] * x_line
        ax.plot(x_line, y_line, 'r-', linewidth=2, label="GD Step")

        ax.set_title(f"Gradient Descent (Step {i+1}/{len(b0_hist)})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()

        # Convert this frame to image buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=80)
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)

    # Save all frames into GIF
    gif_buf = io.BytesIO()
    frames[0].save(
        gif_buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0
    )
    gif_buf.seek(0)
    return gif_buf


# --------------------------------------------------
# Stable Gradient Descent with Normalization
# --------------------------------------------------
def gradient_descent_normalized(x, y, lr=0.01, n_iters=1000):
    n = len(x)
    b0, b1 = 0.0, 0.0

    b0_hist, b1_hist, cost_hist = [], [], []

    for i in range(n_iters):
        y_pred = b0 + b1 * x
        error = y - y_pred

        cost = (1 / (2 * n)) * np.sum(error ** 2)

        # Stop if diverging
        if np.isnan(cost) or cost > 1e8:
            break

        db0 = -(1 / n) * np.sum(error)
        db1 = -(1 / n) * np.sum(error * x)

        db0 = np.clip(db0, -1e5, 1e5)
        db1 = np.clip(db1, -1e5, 1e5)

        b0 -= lr * db0
        b1 -= lr * db1

        b0_hist.append(b0)
        b1_hist.append(b1)
        cost_hist.append(cost)

    return np.array(b0_hist), np.array(b1_hist), np.array(cost_hist)


# --------------------------------------------------
# Cost Surface Plot
# --------------------------------------------------
def plot_cost_surface(b0_hist, b1_hist, x, y):
    b0_vals = np.linspace(min(b0_hist)-1, max(b0_hist)+1, 50)
    b1_vals = np.linspace(min(b1_hist)-1, max(b1_hist)+1, 50)

    B0, B1 = np.meshgrid(b0_vals, b1_vals)
    J = np.zeros_like(B0)

    for i in range(B0.shape[0]):
        for j in range(B0.shape[1]):
            y_pred = B0[i, j] + B1[i, j] * x
            J[i, j] = np.mean((y - y_pred)**2) / 2

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.contour(B0, B1, J, levels=30)
    ax.plot(b0_hist, b1_hist, 'r.-', label="GD Path")
    ax.set_xlabel("b₀")
    ax.set_ylabel("b₁")
    ax.set_title("Gradient Descent Path on Cost Surface")
    ax.legend()
    return fig


# --------------------------------------------------
# STREAMLIT APP
# --------------------------------------------------
def main():

    st.title("📊 Advanced Linear Regression Learning Lab")
    st.write("Upload → Select Features → Train → Visualize → Predict")

    # ---------------------------
    # Upload Data
    # ---------------------------
    st.header("1️⃣ Upload Training Excel File")
    uploaded = st.file_uploader("Upload Excel (.xlsx/.xls)", type=["xlsx", "xls"])

    if uploaded is None:
        return

    df = pd.read_excel(uploaded)
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least 2 numeric columns.")
        return

    # ---------------------------
    # Select Features
    # ---------------------------
    st.header("2️⃣ Select Feature & Target")
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Feature (X)", numeric_cols)
    with col2:
        y_col = st.selectbox("Target (Y)", [c for c in numeric_cols if c != x_col])

    data = df[[x_col, y_col]].dropna()
    x_raw = data[x_col].astype(float).values
    y_raw = data[y_col].astype(float).values

    # ---------------------------
    # Normalize
    # ---------------------------
    x_mean, x_std = x_raw.mean(), x_raw.std()
    y_mean, y_std = y_raw.mean(), y_raw.std()

    x = (x_raw - x_mean) / x_std
    y = (y_raw - y_mean) / y_std

    # ---------------------------
    # Train Model
    # ---------------------------
    st.header("3️⃣ Train Linear Regression Model")
    lr = st.slider("Learning Rate", 0.0001, 1.0, 0.01)
    n_iters = st.slider("Iterations", 50, 5000, 500)

    if st.button("🚀 Train Model"):
        b0_hist, b1_hist, cost_hist = gradient_descent_normalized(x, y, lr, n_iters)

        if len(cost_hist) == 0:
            st.error("Gradient Descent diverged. Try lowering learning rate.")
            return

        st.session_state.update({
            "b0_hist": b0_hist,
            "b1_hist": b1_hist,
            "cost_hist": cost_hist,
            "x_raw": x_raw,
            "y_raw": y_raw,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std
        })

        st.success("Training Complete! 🎉")

    # ---------------------------
    # If model exists
    # ---------------------------
    if "b0_hist" in st.session_state:

        b0_hist = st.session_state["b0_hist"]
        b1_hist = st.session_state["b1_hist"]
        cost_hist = st.session_state["cost_hist"]

        # ---------------------------
        # Cost Curve
        # ---------------------------
        st.header("4️⃣ Cost Function Analysis")
        fig, ax = plt.subplots()
        ax.plot(cost_hist)
        ax.set_title("Cost Reduction Over Iterations")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost J")
        st.pyplot(fig)

        # ---------------------------
        # GD Surface Plot
        # ---------------------------
        st.header("5️⃣ Gradient Descent Surface Visualization")
        fig2 = plot_cost_surface(
            b0_hist, b1_hist,
            (x_raw - x_mean) / x_std,
            (y_raw - y_mean) / y_std
        )
        st.pyplot(fig2)

        # ---------------------------
        # GD Animation
        # ---------------------------
        st.header("🎞️ Gradient Descent Line Animation")
        with st.spinner("Creating animation..."):
            gif_buf = animate_gd(x_raw, y_raw, b0_hist, b1_hist)
        st.image(gif_buf)

        # ---------------------------
        # Convert Back to Original Scale
        # ---------------------------
        b0_norm = b0_hist[-1]
        b1_norm = b1_hist[-1]

        final_b1 = (b1_norm * y_std) / x_std
        final_b0 = y_mean + y_std * b0_norm - final_b1 * x_mean

        st.success(f"📌 Final Model:  ŷ = {final_b0:.4f} + {final_b1:.4f} × X")

        # Predictions
        y_pred = final_b0 + final_b1 * x_raw

        # ---------------------------
        # Performance Metrics
        # ---------------------------
        st.header("6️⃣ Performance Metrics")

        n = len(y_pred)
        r2 = r2_score(y_raw, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2)

        st.json({
            "MAE": mean_absolute_error(y_raw, y_pred),
            "MSE": mean_squared_error(y_raw, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_raw, y_pred)),
            "R²": r2,
            "Adjusted R²": adj_r2
        })

        # ---------------------------
        # Prediction on New File
        # ---------------------------
        st.header("7️⃣ Upload New File for Predictions")
        pred_file = st.file_uploader("Upload Excel for Prediction", type=["xlsx", "xls"], key="predict")

        if pred_file:
            pred_df = pd.read_excel(pred_file)

            if x_col not in pred_df.columns:
                st.error(f"Column '{x_col}' not found in uploaded dataset.")
            else:
                pred_df["Predicted_Y"] = final_b0 + final_b1 * pred_df[x_col]
                st.dataframe(pred_df.head())

                st.download_button(
                    "Download Predictions",
                    pred_df.to_csv(index=False),
                    "predictions.csv",
                    "text/csv"
                )


# --------------------------------------------------
# RUN APP
# --------------------------------------------------
if __name__ == "__main__":
    main()

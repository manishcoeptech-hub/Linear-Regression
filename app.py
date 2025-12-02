import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib.animation import FuncAnimation
import io

def animate_gd(x, y, b0_hist, b1_hist):
    fig, ax = plt.subplots(figsize=(6,4))

    ax.scatter(x, y, color='blue', label='Data points')
    line, = ax.plot([], [], 'r-', linewidth=2, label="Model")
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Gradient Descent Line Movement (Animation)")

    x_line = np.linspace(np.min(x), np.max(x), 100)

    def update(frame):
        b0 = b0_hist[frame]
        b1 = b1_hist[frame]
        y_line = b0 + b1 * x_line
        line.set_data(x_line, y_line)
        return line,

    anim = FuncAnimation(fig, update, frames=len(b0_hist), interval=60)

    # Save animation to GIF in memory buffer
    buf = io.BytesIO()
    anim.save(buf, writer='pillow', format='gif')
    buf.seek(0)
    plt.close(fig)
    return buf

# --------------------------------------------------
# Gradient Descent (Stable Version with Normalization)
# --------------------------------------------------
def gradient_descent_normalized(x, y, lr=0.01, n_iters=1000):
    n = len(x)
    b0, b1 = 0.0, 0.0

    b0_hist, b1_hist, cost_hist = [], [], []

    for i in range(n_iters):

        y_pred = b0 + b1 * x
        error = y - y_pred

        cost = (1 / (2 * n)) * np.sum(error ** 2)

        # Safety check for divergence
        if np.isnan(cost) or cost > 1e8:
            break

        db0 = -(1 / n) * np.sum(error)
        db1 = -(1 / n) * np.sum(error * x)

        # Gradient clipping to prevent explosion
        db0 = np.clip(db0, -1e5, 1e5)
        db1 = np.clip(db1, -1e5, 1e5)

        b0 -= lr * db0
        b1 -= lr * db1

        b0_hist.append(b0)
        b1_hist.append(b1)
        cost_hist.append(cost)

    return np.array(b0_hist), np.array(b1_hist), np.array(cost_hist)


# --------------------------------------------------
# Cost Function Surface Plot
# --------------------------------------------------
def plot_cost_surface(b0_hist, b1_hist, x, y):
    b0_vals = np.linspace(min(b0_hist)-1, max(b0_hist)+1, 50)
    b1_vals = np.linspace(min(b1_hist)-1, max(b1_hist)+1, 50)

    B0, B1 = np.meshgrid(b0_vals, b1_vals)
    J = np.zeros_like(B0)

    for i in range(B0.shape[0]):
        for j in range(B0.shape[1]):
            y_pred = B0[i, j] + B1[i, j] * x
            J[i, j] = np.mean((y - y_pred) ** 2) / 2

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.contour(B0, B1, J, levels=30)
    ax.plot(b0_hist, b1_hist, 'r.-', label="Gradient Descent Path")
    ax.set_xlabel("b₀")
    ax.set_ylabel("b₁")
    ax.legend()
    ax.set_title("Gradient Descent Path on Cost Surface")
    return fig
# ------------------------------------
# GD ANIMATION
# ------------------------------------
st.header("🎞️ Gradient Descent Animation of Regression Line")

x_raw = st.session_state["x_raw"]
y_raw = st.session_state["y_raw"]

with st.spinner("Generating animation..."):
    anim_buf = animate_gd(
        x_raw,
        y_raw,
        st.session_state["b0_hist"],
        st.session_state["b1_hist"]
    )

st.image(anim_buf, caption="Gradient Descent Line Convergence Animation")


# --------------------------------------------------
# Streamlit App
# --------------------------------------------------
def main():

    st.title("📊 Advanced Linear Regression Learning Lab")
    st.write("Upload → Train → Visualize GD → Evaluate → Predict")

    # -----------------------------
    # Upload Training Data
    # -----------------------------
    st.header("1️⃣ Upload Training Excel File")
    uploaded_file = st.file_uploader("Upload Excel (.xlsx/.xls)", type=["xlsx", "xls"])

    if uploaded_file is None:
        return

    df = pd.read_excel(uploaded_file)
    st.dataframe(df.head())

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns.")
        return

    # -----------------------------
    # Feature Selection
    # -----------------------------
    st.header("2️⃣ Select Feature & Target")
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Feature (X)", numeric_cols)
    with col2:
        y_col = st.selectbox("Target (Y)", [c for c in numeric_cols if c != x_col])

    data = df[[x_col, y_col]].dropna()
    x_raw = data[x_col].values.astype(float)
    y_raw = data[y_col].values.astype(float)

    # -----------------------------
    # Data Normalization (Important Fix)
    # -----------------------------
    x_mean, x_std = np.mean(x_raw), np.std(x_raw)
    y_mean, y_std = np.mean(y_raw), np.std(y_raw)

    x = (x_raw - x_mean) / x_std
    y = (y_raw - y_mean) / y_std

    # -----------------------------
    # Train Model
    # -----------------------------
    st.header("3️⃣ Train Linear Regression Model")
    lr = st.slider("Learning Rate", 0.0001, 1.0, 0.01)
    n_iters = st.slider("Iterations", 50, 5000, 500)

    if st.button("🚀 Train Model"):
        b0_hist, b1_hist, cost_hist = gradient_descent_normalized(x, y, lr, n_iters)

        if len(cost_hist) == 0:
            st.error("Gradient Descent diverged. Try a smaller learning rate.")
            return

        st.success("Training complete! ✓")

        st.session_state["b0_hist"] = b0_hist
        st.session_state["b1_hist"] = b1_hist
        st.session_state["cost_hist"] = cost_hist
        st.session_state["x_raw"] = x_raw
        st.session_state["y_raw"] = y_raw
        st.session_state["x_mean"] = x_mean
        st.session_state["x_std"] = x_std
        st.session_state["y_mean"] = y_mean
        st.session_state["y_std"] = y_std

    # If model exists
    if "b0_hist" in st.session_state:

        b0_hist = st.session_state["b0_hist"]
        b1_hist = st.session_state["b1_hist"]
        cost_hist = st.session_state["cost_hist"]

        st.header("4️⃣ Cost Function Analysis")
        fig, ax = plt.subplots()
        ax.plot(cost_hist)
        ax.set_title("Cost Reduction Over Iterations")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost J")
        st.pyplot(fig)

        st.header("5️⃣ Gradient Descent Optimization Visualization")
        fig2 = plot_cost_surface(b0_hist, b1_hist,
                                 (st.session_state["x_raw"] - st.session_state["x_mean"]) / st.session_state["x_std"],
                                 (st.session_state["y_raw"] - st.session_state["y_mean"]) / st.session_state["y_std"])
        st.pyplot(fig2)

        # -----------------------------
        # Convert back to original scale
        # -----------------------------
        b0_norm = b0_hist[-1]
        b1_norm = b1_hist[-1]

        final_b1 = (b1_norm * st.session_state["y_std"]) / st.session_state["x_std"]
        final_b0 = st.session_state["y_mean"] + st.session_state["y_std"] * b0_norm - final_b1 * st.session_state["x_mean"]

        st.success(f"📌 Final Model:  ŷ = {final_b0:.4f} + {final_b1:.4f} × X")

        # Predictions
        y_pred = final_b0 + final_b1 * st.session_state["x_raw"]

        # -----------------------------
        # Performance Metrics
        # -----------------------------
        st.header("6️⃣ Performance Metrics")

        mae = mean_absolute_error(st.session_state["y_raw"], y_pred)
        mse = mean_squared_error(st.session_state["y_raw"], y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(st.session_state["y_raw"], y_pred)
        n = len(y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2)

        st.json({
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2,
            "Adjusted R²": adj_r2
        })

        # -----------------------------
        # Prediction on New Data
        # -----------------------------
        st.header("7️⃣ Upload Another File for Prediction")
        pred_file = st.file_uploader("Upload Excel to Predict", key="predict", type=["xlsx", "xls"])

        if pred_file:
            pred_df = pd.read_excel(pred_file)

            if x_col not in pred_df.columns:
                st.error(f"Column '{x_col}' is missing in uploaded file.")
            else:
                pred_df["Predicted_Y"] = final_b0 + final_b1 * pred_df[x_col]
                st.dataframe(pred_df.head())

                st.download_button(
                    "Download Predictions",
                    pred_df.to_csv(index=False),
                    "predictions.csv",
                    "text/csv"
                )

if __name__ == "__main__":
    main()

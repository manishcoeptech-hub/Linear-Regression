import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# Helper: Batch Gradient Descent for Simple Linear Regression
# --------------------------------------------------
def gradient_descent(x, y, lr=0.01, n_iters=1000):
    n = len(x)
    b0, b1 = 0.0, 0.0

    b0_hist, b1_hist, cost_hist = [], [], []

    for _ in range(n_iters):
        y_pred = b0 + b1 * x
        error = y - y_pred

        cost = (1 / (2 * n)) * np.sum(error ** 2)
        db0 = -(1 / n) * np.sum(error)
        db1 = -(1 / n) * np.sum(error * x)

        b0 -= lr * db0
        b1 -= lr * db1

        b0_hist.append(b0)
        b1_hist.append(b1)
        cost_hist.append(cost)

    return np.array(b0_hist), np.array(b1_hist), np.array(cost_hist)


# --------------------------------------------------
# Closed-form for comparison
# --------------------------------------------------
def closed_form_solution(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)
    if den == 0:
        return y_mean, 0
    b1 = num / den
    b0 = y_mean - b1 * x_mean
    return b0, b1


# --------------------------------------------------
# Visualize Gradient Descent on Cost Function Space
# --------------------------------------------------
def plot_cost_surface(b0_hist, b1_hist, x, y):
    b0_vals = np.linspace(min(b0_hist)-1, max(b0_hist)+1, 50)
    b1_vals = np.linspace(min(b1_hist)-1, max(b1_hist)+1, 50)

    B0, B1 = np.meshgrid(b0_vals, b1_vals)
    cost_grid = np.zeros_like(B0)

    for i in range(B0.shape[0]):
        for j in range(B0.shape[1]):
            y_pred = B0[i, j] + B1[i, j] * x
            cost_grid[i, j] = np.mean((y - y_pred) ** 2) / 2

    fig, ax = plt.subplots(figsize=(6, 4))
    contour = ax.contour(B0, B1, cost_grid, levels=30)
    ax.plot(b0_hist, b1_hist, 'r.-', label="Gradient Descent Path")
    ax.set_xlabel("b0")
    ax.set_ylabel("b1")
    ax.set_title("Gradient Descent on Cost Function Surface")
    ax.legend()

    return fig


# --------------------------------------------------
# Streamlit App
# --------------------------------------------------
def main():
    st.set_page_config(page_title="Advanced Linear Regression Lab", layout="wide")
    st.title("📊 Linear Regression Complete Learning Lab")

    st.write("Upload a dataset → Train model → Visualize Gradient Descent → Evaluate performance → Predict.")

    st.header("1️⃣ Upload Training Excel File")
    uploaded_file = st.file_uploader("Upload Excel (.xlsx/.xls)", type=["xlsx", "xls"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("### Preview of Training Data")
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("Need at least two numeric columns.")
            return

        st.header("2️⃣ Select Feature & Target")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Feature (X)", numeric_cols)
        with col2:
            y_col = st.selectbox("Target (Y)", [c for c in numeric_cols if c != x_col])

        data = df[[x_col, y_col]].dropna()
        x = data[x_col].values.astype(float)
        y = data[y_col].values.astype(float)

        st.info(f"Selected X = {x_col}, Y = {y_col}")

        st.header("3️⃣ Train Linear Regression Model")
        lr = st.slider("Learning Rate", 0.0001, 1.0, 0.05)
        n_iters = st.slider("Number of Iterations", 50, 5000, 500)

        train_button = st.button("🚀 Train Model")

        if train_button:
            b0_hist, b1_hist, cost_hist = gradient_descent(x, y, lr, n_iters)
            st.session_state["b0_hist"] = b0_hist
            st.session_state["b1_hist"] = b1_hist
            st.session_state["cost_hist"] = cost_hist

            st.success("Training complete!")

        if "b0_hist" in st.session_state:
            b0_hist = st.session_state["b0_hist"]
            b1_hist = st.session_state["b1_hist"]
            cost_hist = st.session_state["cost_hist"]

            st.header("4️⃣ Cost Function Analysis")
            fig1, ax1 = plt.subplots()
            ax1.plot(cost_hist)
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Cost J")
            ax1.set_title("Cost Reduction Over Iterations")
            st.pyplot(fig1)

            st.header("5️⃣ Gradient Descent Optimization Visualization")
            fig2 = plot_cost_surface(b0_hist, b1_hist, x, y)
            st.pyplot(fig2)

            final_b0 = b0_hist[-1]
            final_b1 = b1_hist[-1]
            st.success(f"Final Model:  ŷ = {final_b0:.4f} + {final_b1:.4f} × X")

            # Predictions
            y_pred = final_b0 + final_b1 * x

            # --------------------------------------------------
            # Performance Metrics
            # --------------------------------------------------
            st.header("6️⃣ Performance Metrics")

            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)
            n = len(y)
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2)

            metrics = {
                "Mean Absolute Error (MAE)": mae,
                "Mean Squared Error (MSE)": mse,
                "Root MSE (RMSE)": rmse,
                "R² Score": r2,
                "Adjusted R² Score": adj_r2,
            }

            st.json(metrics)

            # --------------------------------------------------
            # Prediction on New Excel File
            # --------------------------------------------------
            st.header("7️⃣ Upload Another Excel File for Prediction")

            pred_file = st.file_uploader("Upload new Excel for prediction", type=["xlsx", "xls"], key="predict")

            if pred_file:
                pred_df = pd.read_excel(pred_file)
                if x_col not in pred_df.columns:
                    st.error(f"'{x_col}' not found in uploaded file.")
                else:
                    pred_df["Predicted_Y"] = final_b0 + final_b1 * pred_df[x_col]
                    st.write("### Prediction Results")
                    st.dataframe(pred_df.head())
                    st.download_button(
                        label="Download Predictions",
                        data=pred_df.to_csv(index=False),
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()

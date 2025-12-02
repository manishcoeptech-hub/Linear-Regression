import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Helper: Batch Gradient Descent for Simple Linear Regression
# --------------------------------------------------
def gradient_descent(x, y, lr=0.01, n_iters=1000):
    """
    x, y: 1D numpy arrays
    lr: learning rate
    n_iters: number of iterations
    Returns:
        b0_hist, b1_hist, cost_hist
    """
    n = len(x)
    b0 = 0.0
    b1 = 0.0

    b0_hist = []
    b1_hist = []
    cost_hist = []

    for i in range(n_iters):
        y_pred = b0 + b1 * x
        error = y - y_pred

        # Mean Squared Error cost
        cost = (1 / (2 * n)) * np.sum(error ** 2)

        # Gradients
        db0 = -(1 / n) * np.sum(error)
        db1 = -(1 / n) * np.sum(error * x)

        # Parameter update
        b0 = b0 - lr * db0
        b1 = b1 - lr * db1

        # Store history
        b0_hist.append(b0)
        b1_hist.append(b1)
        cost_hist.append(cost)

    return np.array(b0_hist), np.array(b1_hist), np.array(cost_hist)


# --------------------------------------------------
# Helper: Closed-form (Normal Equation) for Comparison
# --------------------------------------------------
def closed_form_solution(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)
    if den == 0:
        return 0.0, y_mean
    b1 = num / den
    b0 = y_mean - b1 * x_mean
    return b0, b1


# --------------------------------------------------
# Streamlit App
# --------------------------------------------------
def main():
    st.set_page_config(page_title="Linear Regression Lab", layout="wide")
    st.title("📊 Linear Regression Learning Lab")
    st.write(
        "Upload a dataset, run **gradient descent**, and visualize "
        "cost, parameters, and convergence step by step."
    )

    # Sidebar: controls
    st.sidebar.header("⚙️ Settings")
    lr = st.sidebar.slider("Learning Rate (α)", 0.0001, 1.0, 0.05, step=0.0001)
    n_iters = st.sidebar.slider("Iterations", 10, 5000, 500, step=10)
    show_details = st.sidebar.checkbox("Show detailed calculations for current step", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Tip:** Start with a small learning rate like 0.01 or 0.05.")

    # File uploader
    st.subheader("1️⃣ Upload your Excel file")
    uploaded_file = st.file_uploader("Upload an Excel file (.xlsx or .xls)", type=["xlsx", "xls"])

    if uploaded_file is None:
        st.info("Please upload an Excel file to begin.")
        return

    # Read Excel
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    st.write("### Preview of Data")
    st.dataframe(df.head())

    # Select columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least two numeric columns (one feature X and one target Y).")
        return

    st.subheader("2️⃣ Select feature (X) and target (Y)")
    col1, col2 = st.columns(2)
    with col1:
        feature_col = st.selectbox("Select feature column (X)", numeric_cols)
    with col2:
        target_col = st.selectbox("Select target column (Y)", [c for c in numeric_cols if c != feature_col])

    # Extract and clean data
    data = df[[feature_col, target_col]].dropna()
    x = data[feature_col].values.astype(float)
    y = data[target_col].values.astype(float)

    st.write(f"Selected **X = {feature_col}**, **Y = {target_col}**")
    st.write(f"Number of valid data points: {len(x)}")

    if len(x) < 2:
        st.error("Not enough data points after cleaning. Need at least 2.")
        return

    # Train button
    st.subheader("3️⃣ Train Linear Regression with Gradient Descent")
    run_button = st.button("🚀 Run Gradient Descent")

    if run_button:
        # Run GD and store in session_state
        b0_hist, b1_hist, cost_hist = gradient_descent(x, y, lr=lr, n_iters=n_iters)
        st.session_state["b0_hist"] = b0_hist
        st.session_state["b1_hist"] = b1_hist
        st.session_state["cost_hist"] = cost_hist
        st.success("Training complete! Use the controls below to explore each step.")

    # If we already have history, show controls/plots
    if "b0_hist" in st.session_state:
        b0_hist = st.session_state["b0_hist"]
        b1_hist = st.session_state["b1_hist"]
        cost_hist = st.session_state["cost_hist"]

        # Slider to choose iteration
        st.subheader("4️⃣ Explore Training Steps")
        max_iter = len(cost_hist) - 1
        iter_idx = st.slider("Iteration number", 0, max_iter, max_iter, step=1)

        current_b0 = b0_hist[iter_idx]
        current_b1 = b1_hist[iter_idx]
        current_cost = cost_hist[iter_idx]

        # Closed-form comparison
        cf_b0, cf_b1 = closed_form_solution(x, y)

        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("b₀ (Intercept)", f"{current_b0:.4f}")
        with colB:
            st.metric("b₁ (Slope)", f"{current_b1:.4f}")
        with colC:
            st.metric("Cost J (MSE/2)", f"{current_cost:.6f}")

        st.caption("Note: Cost function used here is J = (1 / (2n)) * Σ (y - ŷ)²")

        st.markdown("#### Comparison with Closed-form Solution")
        st.write(
            f"- Closed-form b₀: **{cf_b0:.4f}**,  b₁: **{cf_b1:.4f}**  "
            f"(using normal equation / covariance method)"
        )

        # Plots: Cost vs Iterations, and Data + line
        st.subheader("5️⃣ Visualizations")

        plot_col1, plot_col2 = st.columns(2)

        # Cost vs iteration
        with plot_col1:
            st.markdown("**Cost vs Iterations (Convergence)**")
            fig1, ax1 = plt.subplots()
            ax1.plot(range(len(cost_hist)), cost_hist)
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Cost J")
            ax1.set_title("Cost Function During Training")
            ax1.grid(True)
            ax1.axvline(iter_idx, color="gray", linestyle="--", linewidth=1)
            st.pyplot(fig1)

        # Data + regression line
        with plot_col2:
            st.markdown("**Data and Regression Line (for selected iteration)**")
            fig2, ax2 = plt.subplots()
            ax2.scatter(x, y, label="Data points")

            x_line = np.linspace(np.min(x), np.max(x), 100)
            y_line = current_b0 + current_b1 * x_line
            ax2.plot(x_line, y_line, label=f"Line at iter {iter_idx}", linewidth=2)

            ax2.set_xlabel(feature_col)
            ax2.set_ylabel(target_col)
            ax2.set_title("Best-fit Line During Training")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)

        # Detailed calculations for current iteration
        if show_details:
            st.subheader("6️⃣ Detailed Calculations for Selected Iteration")

            # Recompute predictions at this step
            y_pred_iter = current_b0 + current_b1 * x
            error_iter = y - y_pred_iter
            sq_error_iter = error_iter ** 2

            detail_df = pd.DataFrame({
                feature_col: x,
                target_col: y,
                "y_pred (Ŷ)": y_pred_iter,
                "error (y - Ŷ)": error_iter,
                "squared error": sq_error_iter,
            })

            st.write(
                "Below table shows how predictions and errors look at the selected iteration. "
                "You can scroll horizontally if needed."
            )
            st.dataframe(detail_df.head(20))

            st.markdown("**Gradient formulas used:**")
            st.latex(r"\frac{\partial J}{\partial b_0} = -\frac{1}{n}\sum (y_i - \hat{y}_i)")
            st.latex(r"\frac{\partial J}{\partial b_1} = -\frac{1}{n}\sum (y_i - \hat{y}_i)x_i")

            # Show gradient values at this step
            n = len(x)
            db0_iter = -(1 / n) * np.sum(error_iter)
            db1_iter = -(1 / n) * np.sum(error_iter * x)

            grad_col1, grad_col2 = st.columns(2)
            with grad_col1:
                st.write(f"∂J/∂b₀ at iter {iter_idx} = **{db0_iter:.6f}**")
            with grad_col2:
                st.write(f"∂J/∂b₁ at iter {iter_idx} = **{db1_iter:.6f}**")

        # Some theory recap
        st.subheader("7️⃣ Theory Recap (for Students)")

        with st.expander("Show / Hide Theory"):
            st.markdown("""
            **Model equation:**

            \\[
            \\hat{y} = b_0 + b_1 x
            \\]

            **Cost function (Mean Squared Error):**

            \\[
            J(b_0, b_1) = \\frac{1}{2n} \\sum_{i=1}^n (y_i - \\hat{y}_i)^2
            \\]

            **Gradient Descent update rules:**

            \\[
            b_0 := b_0 - \\alpha \\frac{\\partial J}{\\partial b_0}
            \\]
            \\[
            b_1 := b_1 - \\alpha \\frac{\\partial J}{\\partial b_1}
            \\]

            where

            \\[
            \\frac{\\partial J}{\\partial b_0} = -\\frac{1}{n} \\sum (y_i - \\hat{y}_i)
            \\]
            \\[
            \\frac{\\partial J}{\\partial b_1} = -\\frac{1}{n} \\sum (y_i - \\hat{y}_i) x_i
            \\]

            - Learning rate **α** controls step size.
            - As iterations increase, **cost should generally go down**.
            - When cost stops decreasing and parameters stabilize, we say the algorithm has **converged**.
            """)


if __name__ == "__main__":
    main()

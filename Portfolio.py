import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go

def portfolio_optimizer_page():
    st.title("📈 Portfolio Optimizer")

    def fetch_data(tickers, start, end):
        data = yf.download(tickers, start=start, end=end)["Close"]
        return data.dropna()

    def compute_returns_and_cov(data):
        returns = np.log(data / data.shift(1)).dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        return mean_returns, cov_matrix

    def compute_weights(mean_returns, cov_matrix, target_return):
        u = np.ones(len(mean_returns))
        C_inv = np.linalg.inv(cov_matrix)

        m = mean_returns.to_numpy()
        A = u @ C_inv @ u
        B = u @ C_inv @ m
        C = m @ C_inv @ m
        D = A * C - B**2

        lam = (C - B * target_return) / D
        gam = (A * target_return - B) / D

        weights = lam * (C_inv @ u) + gam * (C_inv @ m)
        return weights

    def minimum_variance_portfolio(mean_returns, cov_matrix):
        u = np.ones(len(mean_returns))
        C_inv = np.linalg.inv(cov_matrix)
        weights = (C_inv @ u) / (u @ C_inv @ u)
        min_return = weights @ mean_returns
        min_risk = np.sqrt(weights @ cov_matrix @ weights)
        return weights, min_return, min_risk

    def plot_frontier(mean_returns, cov_matrix, num_points=100):
        returns = np.linspace(mean_returns.min(), mean_returns.max(), num_points)
        risks = []
        weights_list = []

        for r in returns:
            w = compute_weights(mean_returns, cov_matrix, r)
            sigma = np.sqrt(w @ cov_matrix @ w)
            risks.append(sigma)
            weights_list.append(w)

        return returns, np.array(risks), weights_list

    # UI
    st.subheader("📊 Efficient Frontier & Minimum Variance Portfolio")

    tickers = st.multiselect("Choose up to 6 Stocks", 
                             ["AAPL", "GOOGL", "MSFT", "META", "TSLA", "AMZN", "NVDA", "JPM", "WMT", "NFLX"], 
                             default=["AAPL", "MSFT", "GOOGL"])

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))

    rf = st.number_input("Select Risk-Free Rate (e.g. 0.05 for 5%)", value=0.03, step=0.01)

    if len(tickers) < 2:
        st.warning("Please select at least 2 stocks.")
        return

    if st.button("Run Portfolio Analysis"):
        data = fetch_data(tickers, start_date, end_date)

        # Historical Prices Plot
        st.subheader("📊 Historical Stock Prices")
        price_fig = go.Figure()
        for ticker in tickers:
            price_fig.add_trace(go.Scatter(x=data.index, y=data[ticker], mode='lines', name=ticker))
        price_fig.update_layout(title="Stock Prices Over Time", xaxis_title="Date", yaxis_title="Adjusted Closing Price (USD)",
                                height=450, width=800, template='plotly_white')
        st.plotly_chart(price_fig, use_container_width=True)

        mean_returns, cov_matrix = compute_returns_and_cov(data)

        st.subheader("📈 Annualized Mean Returns")
        st.write(mean_returns)

        st.subheader("📉 Annualized Covariance Matrix")
        st.write(cov_matrix)

        target_returns, risks, weights_list = plot_frontier(mean_returns, cov_matrix)
        min_w, min_ret, min_risk = minimum_variance_portfolio(mean_returns, cov_matrix)

        # Capital Market Line
        M = mean_returns.values
        C = cov_matrix.values
        u = np.ones(len(M))
        C_inv = np.linalg.inv(C)

        mar_w = (M - rf * u) @ C_inv / ((M - rf * u) @ C_inv @ u.T)
        mar_mu = M @ mar_w.T
        mar_risk = np.sqrt(mar_w @ C @ mar_w.T)

        slope = (mar_mu - rf) / mar_risk
        intercept = rf

        cml_risks = np.linspace(0, max(risks) * 1.1, 100)
        cml_returns = rf + slope * cml_risks

        # Frontier + CML Plot
        frontier_fig = go.Figure()
        frontier_fig.add_trace(go.Scatter(x=risks, y=target_returns, mode='lines', name='Efficient Frontier', line=dict(color='blue')))
        frontier_fig.add_trace(go.Scatter(x=[min_risk], y=[min_ret], mode='markers', name='Minimum Variance Portfolio', marker=dict(color='red', size=10)))
        frontier_fig.add_trace(go.Scatter(x=[mar_risk], y=[mar_mu], mode='markers', name='Market Portfolio', marker=dict(color='orange', size=10)))
        frontier_fig.add_trace(go.Scatter(x=cml_risks, y=cml_returns, mode='lines', name='Capital Market Line', line=dict(color='green', dash='dash')))

        frontier_fig.update_layout(title="Efficient Frontier with Capital Market Line",
                                   xaxis_title="Risk (Standard Deviation)",
                                   yaxis_title="Return",
                                   height=600, width=1000,
                                   template='plotly_white')
        st.plotly_chart(frontier_fig, use_container_width=False)

        # Minimum Variance Weights
        st.subheader("⚖️ Minimum Variance Portfolio Weights")
        pie_labels = list(mean_returns.index)
        pie_weights = pd.Series(min_w, index=pie_labels)
        st.write(pie_weights)

        pie_fig = go.Figure(data=[go.Pie(labels=pie_weights.index, values=pie_weights.values, hole=0.4)])
        pie_fig.update_layout(title="Minimum Variance Portfolio Allocation", height=400, width=600)
        st.plotly_chart(pie_fig, use_container_width=False)

        # CML Equation
        st.subheader("📐 Capital Market Line Equation")
        st.markdown("**CML Equation:**")
        st.latex(f"y = {slope:.2f}x + {intercept:.2f}")

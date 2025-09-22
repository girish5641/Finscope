# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# import streamlit as st

# def bsm_analysis_simple(K=1, T=1, r=0.05, sigma=0.6, S_values=None, T_values=None):
#     # Black-Scholes-Merton pricing formula
#     def BSM(S, K, T, r, sigma, t):
#         tau = T - t
#         if tau == 0:
#             return max(S - K, 0), max(K - S, 0)
#         d1=(np.log(S/K)+((r+0.5*sigma**2))*tau) *1/(sigma*np.sqrt(tau))
#         d2=(np.log(S/K)+((r-0.5*sigma**2))*tau)*1/(sigma*np.sqrt(tau))
#         call=S*norm.cdf(d1)-K*np.exp(-r*tau)*norm.cdf(d2)
#         put=-S*norm.cdf(-d1)+K*np.exp(-r*tau)*norm.cdf(-d2)
        
#         return call, put

#     if S_values is None:
#         S_values = np.linspace(0.1, 2, 200)
#     if T_values is None:
#         T_values = [0,0.2,0.4,0.6,0.8,1.0]

#     call_prices = []
#     put_prices = []

#     for t in T_values:
#         calls = []
#         puts = []
#         for S in S_values:
#             c, p = BSM(S, K, T, r, sigma, t)
#             calls.append(c)
#             puts.append(p)
#         call_prices.append(calls)
#         put_prices.append(puts)

#     call_prices = np.array(call_prices)
#     put_prices = np.array(put_prices)

#     # Plot helper functions
#     def plot_2d(S, T, prices, title, ylabel):
#         plt.figure(figsize=(8, 5))
#         for i, t in enumerate(T):
#             plt.plot(S, prices[i], label=f"T={t:.2f}")
#         plt.xlabel("Stock Price (S)")
#         plt.ylabel(ylabel)
#         plt.title(title)
#         plt.legend()
#         plt.grid(True)
#         st.pyplot(plt)
#         plt.clf()

#     def plot_3d_scatter(S, T, prices, zlabel, title):
#         from mpl_toolkits.mplot3d import Axes3D  # noqa
#         fig = plt.figure(figsize=(8, 6))
#         ax = fig.add_subplot(111, projection='3d')
#         x, y, z = [], [], []
#         for i in range(len(T)):
#             for j in range(len(S)):
#                 x.append(S[j])
#                 y.append(T[i])
#                 z.append(prices[i][j])
#         ax.scatter(x, y, z, c=z, cmap='viridis')
#         ax.set_xlabel("Stock Price (S)")
#         ax.set_ylabel("Time to Maturity (t)")
#         ax.set_zlabel(zlabel)
#         ax.set_title(title)
#         st.pyplot(fig)
#         plt.clf()

#     def plot_3d_surface(S_mesh, T_mesh, prices, zlabel, title):
#         from mpl_toolkits.mplot3d import Axes3D  # noqa
#         fig = plt.figure(figsize=(8, 6))
#         ax = fig.add_subplot(111, projection='3d')
#         surf = ax.plot_surface(S_mesh, T_mesh, prices, cmap='ocean_r', edgecolor='none')
#         fig.colorbar(surf)
#         ax.set_xlabel("Stock Price (S)")
#         ax.set_ylabel("Time to Maturity (t)")
#         ax.set_zlabel(zlabel)
#         ax.set_title(title)
#         st.pyplot(fig)
#         plt.clf()

#     # Generate meshgrid for surface plots
#     S_mesh, T_mesh = np.meshgrid(S_values, T_values)

#     # Plot everything
#     st.subheader("2D Call Option Prices")
#     plot_2d(S_values, T_values, call_prices, "Call Option Price vs Stock Price", "Price")

#     st.subheader("2D Put Option Prices")
#     plot_2d(S_values, T_values, put_prices, "Put Option Price vs Stock Price", "Price")

#     st.subheader("3D Scatter Plot - Call Option Prices")
#     plot_3d_scatter(S_values, T_values, call_prices, "Price", "Call Option Price Surface")

#     st.subheader("3D Scatter Plot - Put Option Prices")
#     plot_3d_scatter(S_values, T_values, put_prices, "Price", "Put Option Price Surface")

#     st.subheader("3D Surface Plot - Call Option Prices")
#     plot_3d_surface(S_mesh, T_mesh, call_prices, "Price", "Call Option Price Surface")

#     st.subheader("3D Surface Plot - Put Option Prices")
#     plot_3d_surface(S_mesh, T_mesh, put_prices, "Price", "Put Option Price Surface")


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st

def bsm_analysis_simple(K=1, T=1, r=0.05, sigma=0.6, S_values=None, T_values=None):
    # Black-Scholes-Merton pricing formula
    def BSM(S, K, T, r, sigma, t):
        tau = T - t
        if tau == 0:
            return max(S - K, 0), max(K - S, 0)
        d1=(np.log(S/K)+((r+0.5*sigma**2))*tau) *1/(sigma*np.sqrt(tau))
        d2=(np.log(S/K)+((r-0.5*sigma**2))*tau)*1/(sigma*np.sqrt(tau))
        call=S*norm.cdf(d1)-K*np.exp(-r*tau)*norm.cdf(d2)
        put=-S*norm.cdf(-d1)+K*np.exp(-r*tau)*norm.cdf(-d2)
        
        return call, put

    if S_values is None:
        S_values = np.linspace(0.1, 2, 200)
    if T_values is None:
        T_values = [0,0.2,0.4,0.6,0.8,1.0]

    call_prices = []
    put_prices = []

    for t in T_values:
        calls = []
        puts = []
        for S in S_values:
            c, p = BSM(S, K, T, r, sigma, t)
            calls.append(c)
            puts.append(p)
        call_prices.append(calls)
        put_prices.append(puts)

    call_prices = np.array(call_prices)
    put_prices = np.array(put_prices)

    # Plot helper functions
    def plot_2d(S, T, prices, title, ylabel):
        plt.figure(figsize=(8, 5))
        for i, t in enumerate(T):
            plt.plot(S, prices[i], label=f"T={t:.2f}")
        plt.xlabel("Stock Price (S)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
        plt.clf()

    def plot_3d_scatter(S, T, prices, zlabel, title):
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = [], [], []
        for i in range(len(T)):
            for j in range(len(S)):
                x.append(S[j])
                y.append(T[i])
                z.append(prices[i][j])
        ax.scatter(x, y, z, c=z, cmap='viridis')
        ax.set_xlabel("Stock Price (S)")
        ax.set_ylabel("Time to Maturity (t)")
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        st.pyplot(fig)
        plt.clf()

    def plot_3d_surface(S_mesh, T_mesh, prices, zlabel, title):
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(S_mesh, T_mesh, prices, cmap='ocean_r', edgecolor='none')
        fig.colorbar(surf)
        ax.set_xlabel("Stock Price (S)")
        ax.set_ylabel("Time to Maturity (t)")
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        st.pyplot(fig)
        plt.clf()

    # Generate meshgrid for surface plots
    S_mesh, T_mesh = np.meshgrid(S_values, T_values)

    # Plot everything with styled headers
    st.markdown("""
    <style>
    .st-emotion-cache-1kyxreq {
        background: linear-gradient(90deg, #ff6b6b, #ffa3a3);
        padding: 12px;
        border-radius: 8px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("📈 2D Call Option Prices")
    plot_2d(S_values, T_values, call_prices, "Call Option Price vs Stock Price", "Price")

    st.subheader("📉 2D Put Option Prices")
    plot_2d(S_values, T_values, put_prices, "Put Option Price vs Stock Price", "Price")

    st.subheader("✨ 3D Scatter Plot - Call Option Prices")
    plot_3d_scatter(S_values, T_values, call_prices, "Price", "Call Option Price Surface")

    st.subheader("💎 3D Scatter Plot - Put Option Prices")
    plot_3d_scatter(S_values, T_values, put_prices, "Price", "Put Option Price Surface")

    st.subheader("🌊 3D Surface Plot - Call Option Prices")
    plot_3d_surface(S_mesh, T_mesh, call_prices, "Price", "Call Option Price Surface")

    st.subheader("🔥 3D Surface Plot - Put Option Prices")
    plot_3d_surface(S_mesh, T_mesh, put_prices, "Price", "Put Option Price Surface")
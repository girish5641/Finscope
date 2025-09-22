# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from datetime import datetime
# from scipy.stats import norm
# from scipy.optimize import approx_fprime
# import math

# def calculate_f(C, x, K, t, T, r, sigma):
#     if t == T:
#         return max(0, x - K), max(0, K - x)
#     if x <= 0 or K <= 0 or T - t <= 0:
#         return float('inf')
#     d1 = (math.log(x/K) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * math.sqrt(T - t))
#     d2 = d1 - sigma * math.sqrt(T - t)
#     call_price = x * norm.cdf(d1) - K * math.exp(-r * (T - t)) * norm.cdf(d2)
#     return C - call_price

# def calc_derivative(f, var=0, point=[]):
#     args = np.array(point, dtype=float)
#     def wrapped(x):
#         args[var] = x
#         return f(*args)
#     epsilon = 1e-5
#     grad = approx_fprime([point[var]], wrapped, epsilon)
#     return grad[0]

# def newton(epsilon, C, x, K, t, T, r, x0):
#     for _ in range(1000):
#         deno = calc_derivative(calculate_f, 6, [C, x, K, t, T, r, x0])
#         if deno == 0:
#             return -1
#         x1 = x0 - calculate_f(C, x, K, t, T, r, x0) / deno
#         if abs(x1 - x0) <= epsilon:
#             return x1
#         x0 = x1
#     return -1

# def run_option_analysis_dashboard():
#     # st.set_page_config(layout="wide")
#     st.markdown("""
#         <style>
#             .plot-border {
#                 border: 3px solid #4A90E2;
#                 border-radius: 10px;
#                 padding: 10px;
#                 margin-bottom: 20px;
#                 background-color: #f9f9f9;
#             }
#         </style>
#     """, unsafe_allow_html=True)
#     st.title("Option Data Analysis Dashboard")
#     uploaded_file = st.file_uploader("Upload Option CSV File", type="csv")

#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         df.columns = df.columns.str.strip()
#         df = df[['Symbol', 'Date', 'Expiry', 'Option type', 'Strike Price', 'Close', 'Underlying Value']]
#         option_type = df['Option type'].iloc[0] if 'Option type' in df.columns else 'Option'

#         strike, maturity, price = [], [], []
#         for i in range(len(df)):
#             strike.append(df.loc[i]['Strike Price'])
#             d1 = pd.to_datetime(df.loc[i]['Expiry'])
#             d2 = pd.to_datetime(df.loc[i]['Date'])
#             delta = d1 - d2
#             maturity.append(delta.days)
#             price.append(df.loc[i]['Close'])
#         #---subheader for 3d plot ------
#         st.subheader(f"📊 3D VISUALIZATION OF {option_type} PRICE")
#         fig = plt.figure()
#         ax = plt.axes(projection='3d')
#         ax.grid(True)
#         ax.xaxis.pane.set_edgecolor('black')
#         ax.yaxis.pane.set_edgecolor('black')
#         ax.zaxis.pane.set_edgecolor('black')
#         ax.xaxis.pane.set_alpha(1.0)
#         ax.yaxis.pane.set_alpha(1.0)
#         ax.zaxis.pane.set_alpha(1.0)
#         ax.scatter(strike, maturity, price, marker='.')
#         ax.set_xlabel('Strike Price')
#         ax.set_ylabel('Maturity (in days)')
#         ax.set_zlabel(f'{option_type} Price')
#         ax.set_title(f'3D plot for {option_type} Option')
#         st.pyplot(fig, use_container_width=True)

#         #---subheader for 2d plot ------
#         st.subheader(f"📈 2D VISUALIZATION OF {option_type} PRICE")
#         fig, ax = plt.subplots()
#         ax.grid(True)
        
#         ax.scatter(strike, price, marker='.')
#         ax.set_xlabel('Strike price')
#         ax.set_ylabel(f'{option_type} Price')
#         ax.set_title(f'{option_type} Price vs Strike Price')
        
#         st.pyplot(fig, use_container_width=True)
        
        
#         fig, ax = plt.subplots()
#         ax.grid(True)
        
#         ax.scatter(maturity, price, marker='.')
#         ax.set_xlabel('Maturity (days)')
#         ax.set_ylabel(f'{option_type} Price')
#         ax.set_title(f'{option_type} Price vs Maturity')
#         for spine in ax.spines.values():
#             spine.set_edgecolor('black')
#             spine.set_linewidth(1.5)
#         st.pyplot(fig, use_container_width=True)

#         st.subheader("🧮 Implied Volatility Surface")
#         np.random.seed(42)
#         strike_prices, maturities, volatilities = [], [], []

#         for i in range(len(df)):
#             if np.random.rand() <= 0.5:
#                 try:
#                     d1 = datetime.strptime(df.loc[i]['Expiry'], '%d-%b-%Y')
#                     d2 = datetime.strptime(df.loc[i]['Date'], '%d-%b-%Y')
#                 except:
#                     d1 = pd.to_datetime(df.loc[i]['Expiry'])
#                     d2 = pd.to_datetime(df.loc[i]['Date'])
#                 delta = d1 - d2
#                 if (df.loc[i]['Underlying Value'] > 0 and df.loc[i]['Strike Price'] > 0 and df.loc[i]['Close'] > 0 and delta.days > 0):
#                     sigma = newton(1e-6, df.loc[i]['Close'], df.loc[i]['Underlying Value'],
#                                    df.loc[i]['Strike Price'], 0, delta.days / 365, 0.05, 0.6)
#                     if sigma != -1:
#                         strike_prices.append(df.loc[i]['Strike Price'])
#                         maturities.append(delta.days)
#                         volatilities.append(sigma)

#         col1, col2 = st.columns(2)

#         with col1:
#             st.subheader("⚡Implied Volatility vs Strike Price")
#             fig, ax = plt.subplots()
#             ax.grid(True)
            
#             ax.scatter(strike_prices, volatilities, color='blue', marker='.')
#             ax.set_xlabel('Strike price')
#             ax.set_ylabel('Volatility')
#             ax.set_title('Implied Volatility vs Strike Price')
#             for spine in ax.spines.values():
#                 spine.set_edgecolor('black')
#                 spine.set_linewidth(1.5)
#             st.pyplot(fig, use_container_width=True)

#         with col2:
#             st.subheader("📅 Implied Volatility vs Maturity")
#             fig, ax = plt.subplots()
#             ax.grid(True)
            
#             ax.scatter(maturities, volatilities, color='blue', marker='.')
#             ax.set_xlabel('Maturity (days)')
#             ax.set_ylabel('Volatility')
#             ax.set_title('Implied Volatility vs Maturity')
#             for spine in ax.spines.values():
#                 spine.set_edgecolor('black')
#                 spine.set_linewidth(1.5)
#             st.pyplot(fig, use_container_width=True)

#         st.subheader("🌐 3D IMPLIED VOLATILITY SURFACE")
#         fig = plt.figure()
#         ax = plt.axes(projection='3d')
#         ax.grid(True)
        
#         ax.scatter(strike_prices, maturities, volatilities, color='blue', marker='.')
#         ax.set_xlabel('Strike Price')
#         ax.set_ylabel('Maturity (in days)')
#         ax.set_zlabel('Implied Volatility')
#         ax.set_title('3D plot for Implied Volatility')
#         for spine in ax.spines.values():
#             spine.set_edgecolor('black')
#             spine.set_linewidth(1.5)
#         st.pyplot(fig, use_container_width=True) 



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from scipy.stats import norm
from scipy.optimize import approx_fprime
import math

def calculate_f(C, x, K, t, T, r, sigma):
    if t == T:
        return max(0, x - K), max(0, K - x)
    if x <= 0 or K <= 0 or T - t <= 0:
        return float('inf')
    d1 = (math.log(x/K) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * math.sqrt(T - t))
    d2 = d1 - sigma * math.sqrt(T - t)
    call_price = x * norm.cdf(d1) - K * math.exp(-r * (T - t)) * norm.cdf(d2)
    return C - call_price

def calc_derivative(f, var=0, point=[]):
    args = np.array(point, dtype=float)
    def wrapped(x):
        args[var] = x
        return f(*args)
    epsilon = 1e-5
    grad = approx_fprime([point[var]], wrapped, epsilon)
    return grad[0]

def newton(epsilon, C, x, K, t, T, r, x0):
    for _ in range(1000):
        deno = calc_derivative(calculate_f, 6, [C, x, K, t, T, r, x0])
        if deno == 0:
            return -1
        x1 = x0 - calculate_f(C, x, K, t, T, r, x0) / deno
        if abs(x1 - x0) <= epsilon:
            return x1
        x0 = x1
    return -1

def run_option_analysis_dashboard():
    # Custom CSS styling for the entire page
    st.markdown("""
    <style>
        /* Main page styling */
        .main {
            background-color: #f8f9fa;
            padding: 2rem;
        }
        
        /* Title styling */
        .title {
            color: #2c3e50;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }
        
        /* Subheader styling */
        .subheader {
            color: #2980b9;
            font-size: 1.8rem;
            font-weight: 600;
            margin: 1.5rem 0 1rem 0;
            padding: 0.5rem 0;
            border-bottom: 1px solid #eee;
        }
        
        /* File uploader styling */
        .stFileUploader > div {
            border: 2px dashed #3498db;
            border-radius: 10px;
            padding: 2rem;
            background-color: rgba(52, 152, 219, 0.05);
        }
        
        /* Plot containers */
        .plot-container {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            padding: 1rem;
            margin-bottom: 2rem;
            border: 1px solid #e0e0e0;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            border: none;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            background-color: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Column spacing */
        .stColumns>div {
            gap: 1.5rem;
        }
        
        /* Custom 3D plot styling */
        .mpl-3d-plot {
            background-color: white !important;
            border-radius: 10px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Page title with custom class
    st.markdown('<div class="title">Option Data Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # File uploader section
    st.markdown("### Upload your option data file")
    uploaded_file = st.file_uploader("", type="csv", help="Upload a CSV file containing option data")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            df = df[['Symbol', 'Date', 'Expiry', 'Option type', 'Strike Price', 'Close', 'Underlying Value']]
            
            # Get option type for display purposes
            option_type = df['Option type'].iloc[0] if 'Option type' in df.columns else 'Option'
            
            # Process data
            strike, maturity, price = [], [], []
            for i in range(len(df)):
                strike.append(df.loc[i]['Strike Price'])
                d1 = pd.to_datetime(df.loc[i]['Expiry'])
                d2 = pd.to_datetime(df.loc[i]['Date'])
                delta = d1 - d2
                maturity.append(delta.days)
                price.append(df.loc[i]['Close'])
            
            # ---- 3D Visualization ----
            st.markdown(f'<div class="subheader">📊 3D Visualization of {option_type} Price</div>', unsafe_allow_html=True)
            with st.container():
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Customize 3D plot appearance
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor('#e0e0e0')
                ax.yaxis.pane.set_edgecolor('#e0e0e0')
                ax.zaxis.pane.set_edgecolor('#e0e0e0')
                
                # Create the scatter plot
                scatter = ax.scatter(strike, maturity, price, c=price, cmap='viridis', marker='o', s=50, alpha=0.8)
                
                # Add labels and title
                ax.set_xlabel('Strike Price', fontsize=12, labelpad=10)
                ax.set_ylabel('Maturity (days)', fontsize=12, labelpad=10)
                ax.set_zlabel(f'{option_type} Price', fontsize=12, labelpad=10)
                ax.set_title(f'3D Visualization of {option_type} Prices', fontsize=14, pad=20)
                
                # Add colorbar
                cbar = fig.colorbar(scatter, shrink=0.6, aspect=20)
                cbar.set_label('Option Price', rotation=270, labelpad=15)
                
                st.pyplot(fig, use_container_width=True)
            
            # ---- 2D Visualizations ----
            st.markdown(f'<div class="subheader">📈 2D Visualizations</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                with st.container():
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Custom scatter plot
                    scatter = ax.scatter(strike, price, c=maturity, cmap='coolwarm', marker='o', s=50, alpha=0.8)
                    
                    ax.set_xlabel('Strike Price', fontsize=12)
                    ax.set_ylabel(f'{option_type} Price', fontsize=12)
                    ax.set_title(f'{option_type} Price vs Strike Price', fontsize=14)
                    
                    # Add colorbar for maturity
                    cbar = fig.colorbar(scatter, ax=ax)
                    cbar.set_label('Maturity (days)', rotation=270, labelpad=15)
                    
                    st.pyplot(fig, use_container_width=True)
            
            with col2:
                with st.container():
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Custom scatter plot
                    scatter = ax.scatter(maturity, price, c=strike, cmap='plasma', marker='o', s=50, alpha=0.8)
                    
                    ax.set_xlabel('Maturity (days)', fontsize=12)
                    ax.set_ylabel(f'{option_type} Price', fontsize=12)
                    ax.set_title(f'{option_type} Price vs Maturity', fontsize=14)
                    
                    # Add colorbar for strike prices
                    cbar = fig.colorbar(scatter, ax=ax)
                    cbar.set_label('Strike Price', rotation=270, labelpad=15)
                    
                    st.pyplot(fig, use_container_width=True)
            
            # ---- Implied Volatility Analysis ----
            st.markdown('<div class="subheader">🧮 Implied Volatility Analysis</div>', unsafe_allow_html=True)
            
            # Calculate implied volatilities
            strike_prices, maturities, volatilities = [], [], []
            
            with st.spinner('Calculating implied volatilities...'):
                for i in range(len(df)):
                    if np.random.rand() <= 0.5:  # Sample subset for performance
                        try:
                            d1 = datetime.strptime(df.loc[i]['Expiry'], '%d-%b-%Y')
                            d2 = datetime.strptime(df.loc[i]['Date'], '%d-%b-%Y')
                        except:
                            d1 = pd.to_datetime(df.loc[i]['Expiry'])
                            d2 = pd.to_datetime(df.loc[i]['Date'])
                        delta = d1 - d2
                        if (df.loc[i]['Underlying Value'] > 0 and df.loc[i]['Strike Price'] > 0 
                            and df.loc[i]['Close'] > 0 and delta.days > 0):
                            sigma = newton(1e-6, df.loc[i]['Close'], df.loc[i]['Underlying Value'],
                                         df.loc[i]['Strike Price'], 0, delta.days / 365, 0.05, 0.6)
                            if sigma != -1:
                                strike_prices.append(df.loc[i]['Strike Price'])
                                maturities.append(delta.days)
                                volatilities.append(sigma)
            
            if len(volatilities) > 0:
                # 2D Volatility Plots
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.container():
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.grid(True, linestyle='--', alpha=0.7)
                        
                        # Custom scatter plot with regression line
                        scatter = ax.scatter(strike_prices, volatilities, c=maturities, cmap='autumn', 
                                           marker='D', s=60, alpha=0.8)
                        
                        ax.set_xlabel('Strike Price', fontsize=12)
                        ax.set_ylabel('Implied Volatility', fontsize=12)
                        ax.set_title('Implied Volatility vs Strike Price', fontsize=14)
                        
                        # Add colorbar for maturity
                        cbar = fig.colorbar(scatter, ax=ax)
                        cbar.set_label('Maturity (days)', rotation=270, labelpad=15)
                        
                        st.pyplot(fig, use_container_width=True)
                
                with col2:
                    with st.container():
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.grid(True, linestyle='--', alpha=0.7)
                        
                        # Custom scatter plot with regression line
                        scatter = ax.scatter(maturities, volatilities, c=strike_prices, cmap='winter', 
                                           marker='s', s=60, alpha=0.8)
                        
                        ax.set_xlabel('Maturity (days)', fontsize=12)
                        ax.set_ylabel('Implied Volatility', fontsize=12)
                        ax.set_title('Implied Volatility vs Maturity', fontsize=14)
                        
                        # Add colorbar for strike prices
                        cbar = fig.colorbar(scatter, ax=ax)
                        cbar.set_label('Strike Price', rotation=270, labelpad=15)
                        
                        st.pyplot(fig, use_container_width=True)
                
                # 3D Volatility Surface
                st.markdown('<div class="subheader">🌐 3D Implied Volatility Surface</div>', unsafe_allow_html=True)
                with st.container():
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # Customize 3D plot appearance
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.xaxis.pane.fill = False
                    ax.yaxis.pane.fill = False
                    ax.zaxis.pane.fill = False
                    ax.xaxis.pane.set_edgecolor('#e0e0e0')
                    ax.yaxis.pane.set_edgecolor('#e0e0e0')
                    ax.zaxis.pane.set_edgecolor('#e0e0e0')
                    
                    # Create the scatter plot
                    scatter = ax.scatter(strike_prices, maturities, volatilities, 
                                       c=volatilities, cmap='RdYlBu_r', marker='o', s=50, alpha=0.8)
                    
                    # Add labels and title
                    ax.set_xlabel('Strike Price', fontsize=12, labelpad=10)
                    ax.set_ylabel('Maturity (days)', fontsize=12, labelpad=10)
                    ax.set_zlabel('Implied Volatility', fontsize=12, labelpad=10)
                    ax.set_title('3D Implied Volatility Surface', fontsize=14, pad=20)
                    
                    # Add colorbar
                    cbar = fig.colorbar(scatter, shrink=0.6, aspect=20)
                    cbar.set_label('Volatility', rotation=270, labelpad=15)
                    
                    st.pyplot(fig, use_container_width=True)
            else:
                st.warning("Could not calculate implied volatilities for the provided data.")
        
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
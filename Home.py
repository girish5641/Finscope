import streamlit as st
from PIL import Image

def home_page():
    st.markdown("<h1 style='text-align: center;'>📊 Welcome to FinScope!</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray;'>Explore cutting-edge tools for financial analysis, pricing, and forecasting.</h4>", unsafe_allow_html=True)

    image = Image.open("images/finance_dashboard.jpg")
    st.image(image, caption="Smart Investing Starts Here", width=700)  # Adjust width as needed

    col1, col2 = st.columns([1, 2])

    # with col2:
    #     st.markdown(
    #         """
    #         <div style="text-align: left; padding-right: 200px;">
    #             <h3>🚀 Tools Available</h3>
    #             <p style="color: green; font-weight: 600;">🧮 <b>Black-Scholes-Merton Option Pricing</b>: Compute theoretical option prices.</p>
    #             <p style="color: #17a2b8; font-weight: 600;">📈 <b>Portfolio Optimization</b>: Build efficient portfolios using historical data.</p>
    #             <p style="color: #ffc107; font-weight: 600;">📊 <b>Stock Forecasting</b>: Predict future trends using time series models.</p>
    #             <p style="color: #9b59b6; font-weight: 600;">🌐 <b>Implied Volatility Surface</b>: Visualize market expectations of volatility.</p>
    #             <p>Choose a module from the sidebar to get started!</p>
    #         </div>
    #         """,
    #         unsafe_allow_html=True
    #     )
    with col2:
        st.markdown(
            """
            <div style="text-align: left; padding-right: 200px;">
                <h3 style="margin-bottom: 10px;">🚀 Tools Available</h3>
                <p style="color: green; font-weight: 600; line-height: 1.3; margin: 6px 0;">
                    🧮 <b>Black-Scholes-Merton Option Pricing</b>: Compute theoretical option prices.
                </p>
                <p style="color: #17a2b8; font-weight: 600; line-height: 1.3; margin: 6px 0;">
                    📈 <b>Portfolio Optimization</b>: Build efficient portfolios using historical data.
                </p>
                <p style="color: #ffc107; font-weight: 600; line-height: 1.3; margin: 6px 0;">
                    📊 <b>Stock Forecasting</b>: Predict future trends using time series models.
                </p>
                <p style="color: #9b59b6; font-weight: 600; line-height: 1.3; margin: 6px 0;">
                    🌐 <b>Implied Volatility Surface</b>: Visualize market expectations of volatility.
                </p>
                <p style="margin-top: 12px;">Choose a module from the sidebar to get started!</p>
            </div>
            """,
            unsafe_allow_html=True
    )

   

    st.markdown("---")
    st.markdown("#### 💡 Did You Know?")
    st.markdown("> The Black-Scholes Model revolutionized the options market in the 1970s and earned a Nobel Prize for its creators.")

    st.markdown("### 📽 Finance in Motion")
    st.video("https://www.youtube.com/watch?v=p7HKvqRI_Bo")  # Working video link

    st.markdown("---")
    st.caption("© 2025 FinScope | Created with ❤️ using Streamlit")

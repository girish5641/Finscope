# 📊 Finscope - Financial Data & Portfolio Analytics Dashboard

**Finscope** is a powerful and interactive financial analysis dashboard built with **Streamlit**. It allows users to analyze stock data, explore portfolio optimization strategies, and visualize key financial metrics using a user-friendly web interface.

---

## 🧠 Features

- 📈 **Stock Price Analysis**  
  Fetch and visualize historical stock data using `yfinance` with various interactive plots.

- 📊 **Portfolio Optimization**  
  - Minimum variance portfolio  
  - Efficient frontier plotting  
  - Capital market line (CML) visualization  
  - Allocation pie charts  

- 📐 **Options Analysis**  
  Analyze option pricing using the Black-Scholes-Merton (BSM) model with adjustable parameters.

- 📷 **Visualizations**  
  Built with `matplotlib` and `plotly` for static and interactive plotting, including 3D visualizations.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Data**: yFinance, Pandas, NumPy  
- **Plotting**: Matplotlib, Plotly  
- **Math/Stats**: SciPy  
- **Image Handling**: Pillow  

## Dependencies
Make sure you have the following libraries installed:

  - streamlit
  - numpy
  - pandas
  - matplotlib
  - scipy
  - plotly
  - yfinance
  - Pillow

## Usage
1.Clone the repository
  ```bash
  git clone <repository-url>
  cd <repository-directory>
  ```
2.Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3.Run the Streamlit App
  ```bash
  streamlit run app.py
  ```

## Deployed Link
[![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-brightgreen?logo=streamlit)](https://finscopeapp.streamlit.app/)


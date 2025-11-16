📊 Indian Stock Market News Sentiment & Price Movement Prediction
Using NLP (FinBERT) + Technical Indicators + Machine Learning

Predict NIFTY50 & Top Indian Stock Movements using News Headlines + Price Data

⭐ Project Overview

This project builds an end-to-end Machine Learning system that predicts next-day stock price movement for Indian stocks by combining:

✅ News Sentiment Analysis (FinBERT)
✅ Technical Indicators
✅ Historical Stock Prices (NSE)
✅ ML Models (LightGBM, XGBoost)
✅ Backtesting & Accuracy Evaluation
✅ Interactive Streamlit Dashboard

This project demonstrates real-world skills used in fintech, quant trading, and AI-driven investment systems.

🚀 Features
1. Automated News Scraping

Scrapes Indian finance news from:

Moneycontrol

Economic Times Markets

LiveMint

Financial Express

Yahoo Finance (India)

2. Sentiment Analysis (FinBERT)

Converts raw news headlines into Positive / Negative / Neutral sentiment.

Computes daily sentiment score.

Uses LLM-powered embeddings for better predictive accuracy.

3. Technical Indicators

Includes over 20 indicators:

RSI, MACD, Bollinger Bands

SMA, EMA, VWAP

Volatility, Momentum

Volume Oscillators

4. ML Prediction Model

Models used:
✔ LightGBM
✔ XGBoost
✔ Logistic Regression
✔ Random Forest

Predicts next-day price direction (Up/Down).

5. Backtesting Engine

Simulates trading based on predictions.

Calculates:

Win Rate

Accuracy

Sharpe Ratio

Profit Curve

6. Streamlit Dashboard

📈 Visualizes:

Sentiment timeline

Stock prediction

Backtest results

Live market overview

🏗 Project Structure
indian-stock-sentiment-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_explore_data.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_backtest_evaluation.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── sentiment/
│   ├── models/
│   └── utils/
│
├── models/
│
├── app/           # FastAPI backend
│
└── streamlit_app/ # UI dashboards

🔍 How It Works (Pipeline)
1️⃣ Collect Market Data

Uses yfinance to download NSE historical price data.

2️⃣ Scrape News & Preprocess Text

Clean & tokenize headlines.

3️⃣ Run Sentiment Analysis

FinBERT → sentiment score per headline → aggregated daily score.

4️⃣ Compute Technical Indicators

RSI, MACD, EMA … added as new features.

5️⃣ Train ML Model

Predicts next-day Price Up/Down.

6️⃣ Backtest Model

Evaluates prediction quality over time.

7️⃣ Interactive Dashboard

Streamlit shows sentiment trends, predictions, and backtest results.

⚙️ Installation
1. Clone Repository
git clone https://github.com/YOUR_USERNAME/indian-stock-sentiment-prediction.git
cd indian-stock-sentiment-prediction

2. Install Dependencies
pip install -r requirements.txt

▶️ Run Project
Run FastAPI Backend
uvicorn app.main:app --reload

Run Streamlit Dashboard
streamlit run streamlit_app/Home.py

📈 Sample Outputs
✔ Predicted Price Direction

✔ Sentiment Trend Chart
✔ Backtest Profit Curve
✔ Feature Importance Plot
✔ Live News Sentiment Score

(Images will be added after model training.)

📦 Models Used
Component	Model / Framework
Sentiment Analysis	FinBERT / Transformers
Technical Indicators	TA Library
ML Classifiers	LightGBM, XGBoost
Backend API	FastAPI
Dashboard	Streamlit
🧠 Skills Demonstrated
Data Science & ML

Feature Engineering

Model Building (LGBM, XGBoost)

Backtesting & Evaluation

Time Series Analysis

NLP

News Scraping

Text Cleaning

Transformer-Based Sentiment Models

End-to-End Engineering

Modular Python Code

API Development

Dashboard Creation

Git + Clean Project Structure

📝 Future Enhancements

Real-time streaming news sentiment

LSTM/Transformer price prediction

Option chain sentiment

Multi-stock portfolio modeling

Reinforcement learning trading agent

👨‍💻 Author

Bhavesh Makwana
Data Science & ML Enthusiast (India)
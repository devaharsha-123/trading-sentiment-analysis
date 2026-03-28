# Trading Sentiment Analysis (Hyperliquid × Fear & Greed Index)

## 📊 Overview

This project analyzes 200,000+ trades to understand how market sentiment impacts trading performance.

## 🎯 Objective

To explore the relationship between trader performance and market sentiment, uncover hidden patterns, and derive actionable trading strategies.

## ⚙️ Methodology

1. Load trading and sentiment datasets
2. Clean and preprocess data
3. Merge datasets on date
4. Compute performance metrics (PnL, win rate, profit factor)
5. Analyze sentiment vs performance
6. Visualize insights

## 🔍 Key Insights

* Traders behave **contrarian to sentiment**
* Closing short positions during **Fear yields highest profit (~$207 avg per trade)**
* Extreme Greed has highest win rate but highest risk
* Weak correlation between sentiment and PnL (r ≈ -0.08)

## 📈 Visualizations

* Sentiment vs PnL overview
* Direction vs sentiment heatmap
* Monthly PnL trends
* Account-level analysis
* PnL distribution

## 🧠 Strategy Implications

* Buy during Fear
* Close shorts in Fear
* Reduce exposure during Greed

## 🚀 How to Run

```bash
python trading_sentiment_analysis.py
```

## 📂 Outputs

Charts are saved in the `outputs/` folder.

# attention-and-returns 

> Forecasting financial markets with attention-based deep learning

## Authors

Nai Hola, Enrique Diaz de Leon Hicks, Zac Sardi-Santos  
CS2090B — Harvard University

## Overview

This project explores the use of Transformer architectures to forecast the distribution of 5-day log returns of IBM stock. We combine macroeconomic signals, technical indicators, and market indices to construct a predictive model capable of generating quantile forecasts, and we evaluate its utility via a custom-built trading simulator.

## Problem Statement

1. Can a Transformer, trained on historical financial time series data, predict the 10th, 50th, and 90th percentiles of IBM's 5-day log returns more accurately than a quantile regression baseline?
2. Can these quantile predictions be translated into profitable and risk-aware trading strategies?

## Data

- **Target:** Quantiles of 5-Day Log Returns of IBM Stock
- **Features (350+):**
  - Macroeconomic indicators: CPI, unemployment, inflation
  - Market indices: XLF, XLK, XLE
  - Technical indicators: RSI, MACD, 20-day SMA
  - IBM volume and price
- **Period:** 1998–2025 (daily frequency)

## Key Methodologies

### 📊 Exploratory Data Analysis

- **Multicollinearity Reduction:** Removed redundant OHLCV features; retained Close prices only
- **Normalization:** Applied log-transforms and `StandardScaler` to handle outliers and skewed distributions
- **Windowing Strategy:** Used 30-day rolling sequences to capture temporal signals

### Transformer Model

- **Architecture:**
  - 30-day rolling input windows
  - Embedding projection → Transformer stack (4-head attention) → GlobalAvgPooling → Output Dense layer
  - Output: 3 quantiles (10th, 50th, 90th percentiles)
- **Custom Pinball Loss Function:** For training on quantile regression objectives
- **Training:**
  - 100 epochs, batch size 64, early stopping, learning rate decay

### Evaluation & Results

- **Baseline Pinball Loss:** ~0.0689  
- **Transformer Pinball Loss:** ~0.0091  
- **Improvement:** ~87% reduction in pinball loss

- **Generalization:** Low train-validation gap, <8% MAE difference
- **Simulation:** Transformer model yielded better performance in short-horizon trading, validating attention-based quantile forecasts for financial use

## 📈 Trading Simulator

We built a custom trading simulator to test if the predicted quantile ranges could be used to drive trading decisions. While performance gains diminished over longer horizons, our results were promising for short-term strategies.

## Future Work

- Implement rolling window retraining to adapt to evolving market regimes
- Scale the framework to multi-asset portfolios
- Explore causal modeling to improve robustness to exogenous shocks





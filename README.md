# IPL AI Model - Dream11 Team Predictor

This project predicts optimal Dream11 player points using Machine Learning based on IPL data.  
It combines historical + 2024 match statistics with real-time scraping from ESPNcricinfo to recommend the best playing XI.

# Features

-  XGBoost models to estimate batting and bowling points
-  Integrates actual match averages and dynamically scraped data
-  Real-time scraping from ESPNcricinfo using Selenium
-  Applies custom constraints to select valid playing 11

# How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt

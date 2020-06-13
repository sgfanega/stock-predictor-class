# StockPredictor
A Simple Stock Predictor Class

How to use: 

import class

stock = StockPredictor("ticker_symbol", forecast_days, "machine_learning_type")

To find ticker symbols: https://www.marketwatch.com/tools/quotes/lookup.asp
Forecast days is amount in the future, the less, the more accurate (duh!)
Machine Learning type has two types: Linear Regression ("LR") or Support Vector Machine ("SVM"). Both are great tools to predict.

You will get an array of two things: Array of Prediction (Amount of days is the size of array), and the Confidence.

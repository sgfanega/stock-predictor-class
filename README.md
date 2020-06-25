# StockPredictor
A Simple Stock Predictor Class

How to use: 

CLI: _python StockPredictor.py "AMZN" 20 "LR"_

Ticker Symbol, Forecast Days, and Machine Learning Type

To find ticker symbols: https://www.marketwatch.com/tools/quotes/lookup.asp

Forecast days is amount in the future, the less, the more accurate (duh!)

Machine Learning type has two types: Linear Regression ("LR") or Support Vector Machine ("SVM"). 
Both are great tools to predict.

You will get JSON of two things: The Confidence, and an array of Prediction (Amount of days is the size of array).

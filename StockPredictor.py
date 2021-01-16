#!/usr/bin/python3

# Dependencies
import numpy as np
import pandas as pd
import simplejson as json
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None

class StockPredictor:
    def __init__(self, ticker_symbol = 'MSFT', prediction_days = 10, machine_learning_type = 'LR'):
        # Constructor
        # On default it will use Microsoft, 10 Days, and Linear Regression
        self.ticker_symbol = ticker_symbol
        self.prediction_days = prediction_days
        self.machine_learning_type = machine_learning_type
        self.confidence = ''
        self.predictions = ''
        self.__start_process()
    
    def get_json(self):
        # Gets confidence and predictions and formats it into a JSON
        json = {}

        day_index = 1

        for prediction in self.predictions:
            json["{} {}".format("Day", day_index)] = prediction
            day_index += 1

        json['Confidence'] = self.confidence

        return json
    
    def __start_process(self):
        # Method that downloads the stock data, takes the dependent and independent data, train the data, and predict the data
        self.__check_parameters()
        
        data = self.__get_data()
        x = self.__get_independent_set(data)
        y = self.__get_dependent_set(data)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

        x_forecast = self.__get_independent_forecast(data)

        if self.machine_learning_type == 'LR':
            self.__linear_regression_process(x_train, x_test, y_train, y_test, x_forecast)
        else:
            self.__support_vector_machine_process(x_train, x_test, y_train, y_test, x_forecast)

    def __check_parameters(self):
        # Raises ValueError if prediction_days is not a number/int or ticker_symbol has more than 5 characters
        if type(self.prediction_days) != int:
            raise ValueError('Second Parameter must be a number')
        if len(self.ticker_symbol) > 5:
            raise Exception('First Parameter must be 5 or fewer characters')
        
    def __get_data(self):
        # Gets the data from Yahoo! Finance. Will return an error if the Ticker Symbol is non-NYSE
        # Return 2D List
        try:
            data = yf.download(self.ticker_symbol)
            data = data[['Adj Close']]
            data['Prediction'] = data[['Adj Close']].shift(-self.prediction_days)

            return data
        except:
            print('An error occured while retreiving data from yfinance: You might have used a non-NYSE Ticker Symbol')

    def __get_independent_set(self, data):
        # Gets Independent Set from Data retrieved from Yahoo! Finance
        # Return 2D List
        x = np.array(data.drop(['Prediction'], 1)) # Creates a numpy array without 'Prediction'
        x = x[:-self.prediction_days] # Removes the last N rows
        
        return x

    def __get_dependent_set(self, data):
        # Gets Dependent Set from Data retrieved from Yahoo! Finance
        # Return List
        y = np.array(data['Prediction']) # Creates a numpy array just of 'Prediction'
        y = y[:-self.prediction_days] # Removes the last N rows

        return y

    def __get_independent_forecast(self, data):
        # Gets Independent Forecast from Data retrieved from Yahoo! Finance
        # Return List
        x_forecast = np.array(data.drop(['Prediction'], 1))[-self.prediction_days:]
        
        return x_forecast
    
    def __linear_regression_process(self, x_train, x_test, y_train, y_test, x_forecast):
        # Uses Linear Regression to predict forecast
        lr = LinearRegression()
        lr.fit(x_train, y_train)

        self.confidence = lr.score(x_test, y_test)
        self.predictions = lr.predict(x_forecast)
    
    def __support_vector_machine_process(self, x_train, x_test, y_train, y_test, x_forecast):
        # Uses Support Vector Regression to predict forecast
        svr = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1)
        svr.fit(x_train, y_train)

        self.confidence = svr.score(x_test, y_test)
        self.predictions = svr.predict(x_forecast)

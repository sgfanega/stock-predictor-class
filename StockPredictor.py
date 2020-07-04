# This program will take stock data from Yahoo! Finance and use Machine Learning to Predict future stock prices
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Dependencies
import yfinance as yf
import numpy as np
import pandas as pd
import sys
import simplejson as json
pd.options.mode.chained_assignment = None


class StockPredictor:
    # Constructor
    def __init__(self, ticker_symbol, forecast_days=10, machine_learning_type="LR"):
        self.ticker_symbol = ticker_symbol
        self.forecast_days = forecast_days
        self.machine_learning_type = machine_learning_type
        self.confidence = ""
        self.predictions = ""
        self.start_process()

    # Get Method
    # Return JSON
    def get_json(self):
        json_output = {"confidence": self.confidence}
        day_index = 1

        for prediction in self.predictions:
            json_output["{} {}".format("day", day_index)] = prediction
            day_index += 1

        return json_output

    # Start Method
    def start_process(self):
        # Throws Error if the Machine Learning Type is not LR or SVM
        if self.machine_learning_type != 'LR' and self.machine_learning_type != 'SVM':
            raise ValueError("Machine Learning Type must be either LR or SVM")
        # Throws Error if Forecast Days is not an int
        if type(self.forecast_days) != int:
            raise ValueError("Forecast Days is not a valid amount")

        data = self.__get_data()  # Original Data
        x = self.__get_independent_set(data)  # Independent Data set
        y = self.__get_dependent_set(data)  # Dependent Data set

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # Create the training and testing

        x_forecast = self.__get_independent_forecast(data)

        if self.machine_learning_type == 'LR':
            self.__linear_regression_process(x_train, x_test, y_train, y_test, x_forecast)
        else:
            self.__support_vector_machine_process(x_train, x_test, y_train, y_test, x_forecast)

    # Get Method
    # Return 2D List
    def __get_data(self):
        data = yf.download(self.ticker_symbol)
        data = data[['Adj Close']]  # Discard everything but the Adj Close Column
        data['Prediction'] = data[['Adj Close']].shift(-self.forecast_days)  # Shift the data N up

        return data

    # Get Method
    # Return 2D List
    def __get_independent_set(self, data):
        x = np.array(data.drop(['Prediction'], 1))  # Creates a numpy array without 'Prediction'
        x = x[:-self.forecast_days]  # Removes the last N rows
        return x

    # Get Method
    # Return List
    def __get_dependent_set(self, data):
        y = np.array(data['Prediction'])  # Creates a numpy array just of 'Prediction'
        y = y[:-self.forecast_days]  # Removes the last N rows
        return y

    # Get Method
    # Return List
    def __get_independent_forecast(self, data):
        x_forecast = np.array(data.drop(['Prediction'], 1))[-self.forecast_days:]
        return x_forecast

    # Linear Regression Method
    def __linear_regression_process(self, x_train, x_test, y_train, y_test, x_forecast):
        lr = LinearRegression()
        lr.fit(x_train, y_train)

        self.confidence = lr.score(x_test, y_test)
        self.predictions = lr.predict(x_forecast)

    # Support Vector Machine Method
    def __support_vector_machine_process(self, x_train, x_test, y_train, y_test, x_forecast):
        svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
        svr.fit(x_train, y_train)

        self.confidence = svr.score(x_test, y_test)
        self.predictions = svr.predict(x_forecast)


# Get Method
# Retrieves JSON
def get_json(ticker_symbol, forecast_days, machine_learning_type):
    stock_predictor = StockPredictor(ticker_symbol, forecast_days, machine_learning_type)

    return stock_predictor.get_json()


if __name__ == '__main__':
    if np.size(sys.argv) != 4:
        raise ValueError('The amount of parameters are {}, you need 4 in total'.format(np.size(sys.argv)))

    print(json.dumps(get_json(sys.argv[1], int(sys.argv[2]), sys.argv[3])))

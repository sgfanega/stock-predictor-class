# StockPredictor

This is a simple stock predictor object using Python and its Data Science libraries which includes Sklearn, Pandas, Numpy, and yFinance.
It will use Linear Regression or Support Vector Regression to predict the data.

## Prerequisites

Python 2.7 minimum, Python 3.x recommended as I created this with Python 3.8 environment.

pip 20.x.x recommended but will work with lower versions.

Python libraries:
- numpy
- pandas
- simplejson
- sklearn (install using scikit-learn)
- yfinance

## Installation
```
pip install simplejson
pip install -U scikit-learn
pip install yfinance
```

## Usage
```
import StockPredictor as StockPredictor

stock_predictor = StockPredictor(ticker_symbol, prediction_days, machine_learning_type)

stock_predictor.get_json()
```

**Ticker Symbol** is a one to five character symbol that represents a company's stock name. You must use a NYSE ticker symbol or it will not work.

**Prediction Days** is the amount of days you would like to predict. I recommend a range from 1-30, no fewer, no more. I would do a hard cap on 30 as well as the prediction will be less accurate.

**Machine Learning Type** is the option to either use Linear Regression ('LR') or Support Vector Regression ('SVR').

By default, the **Ticker Smybol** will be 'MSFT', **Prediction Days** will be '10', and **Machine Learning Type** will be 'LR'.

## Output
You will get a JSON data type when you use ```.get_json``` (*Duh!*).

It will include the predicted data and the confidence of the prediction.

Here is an example:

```
{
  "Day 1": 10
  "Day 2": 12
  "Day 3": 13.4
  "Day 4": 12.7
  "Day 5": 11.5
  "Confidence": 0.923322443
}
```

## Contributing
Pull requests are welcome, but it will be unlikely I will change this repo unless it's a dramatic change. This is to show my understanding of both Python, Object-Oriention, and Data Science.

## License
[MIT](https://choosealicense.com/licenses/mit/)

### View on the website [Link](https://tidkeashok007.github.io/Gold_Price_Data_Analysis/)

# Gold Price Prediction Using Machine Learning 
This Notebook deals with prediction of gold prices. The data contains features regarding the Gold Price Data.

  
## Objectives
* The main goal of this notebook is to build a machine learning system that can predict gold prices based on several other stock prices.
* Obtain data insights using pandas.
* Find the correlation of the other features with GLD (gold) price.
* Predict the GLD (gold) price by splitting the data & evaluate the model.


## ABOUT THE DATA
* **Data Overview:** This data file is a comma-separated value(CSV) file format with 2290 rows and 7 columns. It contains 5 columns which are numerical in datatype and one column in Date format. Clearly, the data shows the value of the variables SPX, GLD, USO, SLV, EUR/USD against the dates in the date column.
* Data Sourse: [Link](https://www.kaggle.com/altruistdelhite04/gold-price-data)
* Data type available: .csv

```python
# Import required libraries
import pandas as pd                                          #Load data & perform basic operations
import numpy as np                                           #Numpy Arrays
import matplotlib.pyplot as plt                              #Matplotlib is a low level graph plotting library in python that serves as a visualization utility.
import seaborn as sns                                        #Seaborn is a library that uses Matplotlib underneath to plot graphs. It will be used to visualize random distributions.
from sklearn.model_selection import train_test_split         #Use to split the original data into training data & test data
from sklearn.ensemble import RandomForestRegressor           #Import Random Forest Regression Model
from sklearn import metrics                                  #Useful for finding performance of model

```


## Data consists of various gold prices for several days in the period of 10 years [Date- MM/DD/YYYY].

* **SPX -** The Standard and Poor's 500, or simply the S&P 500, is a stock market index tracking the performance of 500 large companies listed on stock exchanges in the United States.
* **GLD -** SPDR Gold Shares is part of the SPDR family of exchange-traded funds (ETF) managed and marketed by State Street Global Advisors.
* **USO -** The United States Oil Fund ® LP (USO) is an exchange-traded security whose shares may be purchased and sold on the NYSE Arca.
* **SLV -** The iShares Silver Trust (SLV) is an exchange traded fund (ETF) that tracks the price performance of the underlying holdings in the LMBA Silver Price.
* **EUR/USD -** The Currency Pair EUR/USD is the shortened term for the euro against U.S. dollar pair, or cross for the currencies of the European Union (EU) and the United States (USD). The value of the EUR/USD pair is quoted as 1 euro per x U.S. dollars. For example, if the pair is trading at 1.50, it means it takes 1.5 U.S. dollars to buy 1 euro.


## Correlation

![HeatMap](https://user-images.githubusercontent.com/67102886/129918593-2a5de4b9-b6fb-44b1-8b11-26177e6af892.png)

* Gold (GLD) and silver (SLV) are highly corelated to each other -> 0.9
* Gold (GLD) and Standard and Poor's 500 (SPX) are zero correlation -> 0.0
* Rest features expect gold (GLD) are negative correlated with respect to gold (GLD) -> -0.0 & -0.2


## Checking the distribution of GLD price

![Distribution Plot Of GLD Price](https://user-images.githubusercontent.com/67102886/129919722-eaa87a93-4d1d-43a5-bf25-363c014d8bec.png)


## Splitting the Features and Target
* Traget - GLD (gold) price stock
* Features - Other stocks

```python
# axis = 1 (Columns)
# axis = 0 (Rows)
X = gold_price.drop(["Date", "GLD"], axis = 1)
Y = gold_price["GLD"]
```


## Model Training: Random Forest Regressor

```python
# Training the model
# .fit function used to fit our data to this regressive model
regressor.fit(X_train, Y_train)
```
**R scored error: 0.98**


## Compare the Actual Values & Predicted Values in a Plot

![Actaul Values vs Predicted Values](https://user-images.githubusercontent.com/67102886/130235602-f54f3a82-2018-4fd7-a830-a24cb0ed0130.png)

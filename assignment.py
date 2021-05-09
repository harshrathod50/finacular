# Author: Harsh Rathod
# Email: hrathore50@ymail.com

import investpy
import matplotlib.pyplot as plot
import numpy
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# Get the required dataFrame from investing.com
# via investpy
df = investpy.get_stock_historical_data(stock="tisc", country="india", from_date="01/01/2010", to_date="30/12/2020", as_json=False, order='ascending')
# Detele non-required data from it
df.drop(columns=['Open', 'High', 'Low', 'Volume', 'Currency'], inplace=True)

# Plot the data frame
df.plot(kind='line')
plot.title('TATA Steel Ltd. stock')
plot.xlabel('Dates')
plot.ylabel('Stock prices (Closing)')
plot.show()

# Helper variables
dataSet = df.values
# Training data is from from 2010 to 2019
# There 366 days in year 2020
trainingDataLen = len(dataSet) - 366

# Scaling the data
scaler = MinMaxScaler(feature_range=(0,1))
scaledData = scaler.fit_transform(dataSet)

# Preparing for training the data
trainingData = scaledData[0:int(trainingDataLen), :]
xTrain, yTrain = [], []

for i in range(60, len(trainingData)):
    xTrain.append(trainingData[i-60:i, 0])
    yTrain.append(trainingData[i, 0])

# Convert xTrain and yTrain to proper format
xTrain, yTrain = numpy.array(xTrain), numpy.array(yTrain)

# Reshaping is required to feed the data to
# the model
xTrain = numpy.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1] , 1))

# Prepare the lstm model
lstm = Sequential()
lstm.add(LSTM(100, return_sequences=True, input_shape=(xTrain.shape[1], 1)))
lstm.add(LSTM(100))
lstm.add(Dense(1))

# Compile the model
lstm.compile(loss='mean_squared_error')

# Train the model by feeding it data
lstm.fit(xTrain, yTrain, batch_size=1, epochs=1)

# Create the testing dataset
testData = scaledData[trainingDataLen - 60:, :]
xTest = []
yTest = dataSet[trainingDataLen:, :]
for i in range(60, len(testData)):
    xTest.append(testData[i-60:i, 0])
# Convert to proper format
xTest = numpy.array(xTest)
# Reshaping is required
xTest = numpy.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

# Get the stock predictions of year 2020
predictions = lstm.predict(xTest)
predictions = scaler.inverse_transform(predictions)

# Plot the predictions for 2020 stock prices
train = df[:trainingDataLen]
valid = df[trainingDataLen:]
valid['Predictions'] = predictions

plot.figure(figsize=(16, 6))
plot.title('TATA Steel Ltd. stocks (2020 predictions)')
plot.xlabel('Dates')
plot.ylabel('Closing prices (INR)')
plot.plot(train['Close'])
plot.plot(valid[['Close', 'Predictions']])
plot.show()
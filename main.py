import keras
import numpy as np
import pandas_datareader as web
import math
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

found_flag = False
while not found_flag:
    try:
        stock = input("Enter the ticker symbol. ")
        stocks = web.DataReader(stock, data_source='yahoo')
        found_flag = True
    except:
        print("No such company exists. Try again.")

today = date.today()
print(today)

#get the stock quote
df = web.DataReader(stock, data_source='yahoo', start='2010-01-01', end=today)

#get the number of rows and cols
print(df.shape)

#visulaize closing price
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD', fontsize=18)

#create a new data frame with only the close column
data = df.filter(['Close'])
#convert the df to a numpy array
dataset = data.values
#get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * 0.8) # 80% dataset training

#scale the data to preprocessing transformations to do the normalization before inputing the data to the model
scaler = MinMaxScaler(feature_range=(0,1)) 
scaled_data = scaler.fit_transform(dataset)

#create the training dataset
#create the scaled training dataset
train_data = scaled_data[0:training_data_len, :]
#split the data in x_train and y_train
x_train = [] #training features
y_train = [] #target variables


for i in range(60, len(train_data)):
    #append the past 60 values to the x_train dataset
    x_train.append(train_data[i - 60: i, 0]) # 0-59
    y_train.append(train_data[i, 0]) 
    if i <= 60:
        print(x_train)
        print(y_train)
        print()


#convert the x_train and y_train to numpy array to train the LSTM
x_train, y_train = np.array(x_train), np.array(y_train)

#reshape the data
# because the LSTM model expects the input to be 3-dimensional in form of number of samples, number of time steps
# and number of features
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) 

#build the LSTM model architecture
model = Sequential()
#add a layer to the model
model.add(LSTM(50, return_sequences = True, input_shape=(x_train.shape[1],1)))

model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#compile the model
model.compile(optimizer='adam', loss='mean_squared_error') #optimizer is used to improve upon the loss function and
#the loss function is used to measure how well the model did on training

#train the model
model.fit(x_train, y_train, batch_size=1, epochs=1) 

#create the testing dataset
#create a new array containing scaled values 
test_data = scaled_data[training_data_len - 60: , :]
#create the datasets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :] 
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0]) 

#convert the data into numpy array
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#get the model's predicted price values
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions) 

#evaluate our model using the root mean squared error could also use cross entropy loss but look into that before using
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#visualize
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Closed Price USD', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

#show valid and predicted prices
print(valid)

#get the quote
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')
#create a new data frame
new_df = apple_quote.filter(['Close'])
#get the last 60 day closing price values and then convert the dataframe to the array
last_60_days = new_df[-60:].values
#scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#create an empty list
X_test = []
#append past 60 days
X_test.append(last_60_days_scaled)
#convert the X_test to numpy
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#get the predicted scale price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

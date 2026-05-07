from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


data = pd.read_csv('backend/EURUSDm_M30_2010_2024.csv',sep='\t')

print(data.head())
print(data.info())
print(data.describe())


plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['open'], label='Open Price',color='blue')
plt.plot(data['date'], data['close'], label='Close Price',color='red')
plt.title('EURUSDm Open and Close Prices over time')
plt.legend()
#plt.show()


plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['tick_volume'], label='Tick Volume',color='orange')
plt.title('EURUSDm Tick Volume over time')  
#plt.show()

#drop non-numeric columns
numeric_data = data.select_dtypes(include=["int64", "float64"])

#plot 3 check for correlation between features
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numeric Features')
#plt.show()

data['date']= pd.to_datetime(data['date'])

prediction= data.loc[
    (data['date'] >= '2023-01-01') & (data['date'] <= '2024-01-01')
]

plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['close'],color='blue')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('price over time')



#prepare for the lstm model

stock_close = data.filter(['close'])

dataset = stock_close.values #convert to numpy array

training_data_len = int(np.ceil( len(dataset) * .95 )) #95% of data for training

#preprocessing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[:training_data_len]  #95% of all out data

x_train, y_train = [],[]  #x_train is all the features and the y_train is the target variable

# create a sliding window
for i in range(60, len(training_data)):
    x_train.append(training_data[i-60:i, 0])  #features are the previous 60 days
    y_train.append(training_data[i, 0])  #target is the current day
    
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  #reshape for lstm input

#build the model
model = keras.Sequential()

#first layer
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1)))

#second layer

model.add(keras.layers.LSTM(64, return_sequences=False))

#third layer
model.add(keras.layers.Dense(128, activation='relu'))

#fourth layer
model.add(keras.layers.Dropout(0.5))
#final output layer

model.add(keras.layers.Dense(1))

model.summary()

model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[keras.metrics.RootMeanSquaredError()])



training = model.fit(x_train, y_train, batch_size=32, epochs=20)

#prepare test data

test_data = scaled_data[training_data_len - 60:]
x_test, y_test = [], dataset[training_data_len:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#make a predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)



#plotting the data

train = data[:training_data_len]
test= data[training_data_len:]

test= test.copy()
test['Predictions'] = predictions

plt.figure(figsize=(12, 6))
plt.plot(train['date'], train['close'], label='Train (actual data)',color='blue')
plt.plot(test['date'], test['close'], label='Test (actual data)',color='red')
plt.plot(test['date'], test['Predictions'], label='Predicted Price',color='green')
plt.xlabel('Date')  
plt.ylabel('Close Price')
plt.title('EURUSDm Close Price Prediction')
plt.legend()
plt.show()
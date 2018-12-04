
# coding: utf-8

# In[ ]:


#Load needed libraries
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from haversine import haversine
import tensorflow as tf
from tensorflow import keras
from keras import backend
from keras.callbacks import History 
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from sklearn.model_selection import train_test_split
from keras import optimizers


# In[ ]:


#Import CSV with data
train = pd.DataFrame(pd.read_csv("data.csv"))

#Show me the first rows
train.iloc[0:6]


# In[ ]:


#Remove negative fare amounts
train = train[train.fare_amount>=0]
print('Number of taxi rides:',len(train))

#Remove missing data
print(train.isnull().sum())
train = train.dropna(how = 'any', axis = 'rows')


# In[ ]:


#Splitting pickup_datetime into the date, the time and the timezone
train['date'], train['time'], train['UTC'] = train['pickup_datetime'].str.split(' ').str

#Splitting up date further into year, month and date
train['year'], train['month'], train['day'] = train['date'].str.split('-').str

#From the time, we want to the get the hour in which the pick up happened
train['hour'] = train['time'].astype(str).str[:2]

#Remove columns that we no longer need
train = train.drop(['key','pickup_datetime','UTC'], axis=1)


# In[ ]:


#Formatting date
train['date'] = pd.to_datetime(train['date'])

#Get the days of the week
train['week_day'] = train['date'].dt.weekday_name


# In[ ]:


#Change columns to integers
train[['passenger_count','year','month','day','hour']] = train[['passenger_count','year','month','day','hour']].astype(int)
train[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']] = train[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']].astype(float)


# In[ ]:


#Create new column called rush hour is which it has all the rides taken 7-9am and 6-8pm on weekdays
train['rush_hour'] = np.where(train['hour'].isin(['7','8','9','18','19','20']) & train['week_day'].isin(['Monday','Tuesday','Wednesday','Thursday','Friday']), 1, 0)

#Create new column called weekends that has all the rides that.. well.. were taken on the weekend
train['weekend'] = np.where(train['week_day'].isin(['Saturday','Sunday']), 1, 0)

#Taking during the working week
train['workingweek'] = np.where(train['week_day'].isin(['Monday','Tuesday','Wednesday','Thursday','Friday']), 1, 0)

#Delete rows that do not have correct latitutes and longitudes
train = train[(train['pickup_latitude']) > 39 & (train['pickup_latitude'] < 42)]
train = train[(train['dropoff_latitude']) > 39 & (train['dropoff_latitude'] < 42)]
train = train[(train['pickup_longitude']) < -73 & (train['pickup_longitude'] > -75)]
train = train[(train['dropoff_longitude']) < -73 & (train['dropoff_longitude'] > -75)]

#See dataframe so far
train.describe()


# In[ ]:


#Showing a histogram of the fare amount
train[train.fare_amount<100].fare_amount.hist(bins=100)
plt.xlabel('Fare in USD')
plt.show()


# In[ ]:


#Plotting pick ups and drops off- source: https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration/notebook
def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) &            (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) &            (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) &            (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])


def plot_trips(df, BB, figsize=(12, 12), ax=None, c=('r', 'b')):
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    idx = select_within_boundingbox(df, BB)
    ax.scatter(df[idx].pickup_longitude, df[idx].pickup_latitude, c=c[0], s=0.01, alpha=0.5)
    ax.scatter(df[idx].dropoff_longitude, df[idx].dropoff_latitude, c=c[1], s=0.01, alpha=0.5)
    
plot_trips(train, (-74.1, -73.7, 40.6, 40.9))
plot_trips(train, (-74, -73.95, 40.7, 40.8))
plt.show()


# In[ ]:


#Calculating distance from point a to point b straight line
train['straightdistance'] = train.apply(lambda row: haversine((row[2], row[1]), (row[4], row[3])), axis = 1)

#Calculating distance using Manhattan grid
def distance (latitude1, longitude1, latitude2, longitude2):
    distance = haversine((latitude1, longitude1), (latitude2, longitude1)) + haversine((latitude2, longitude1), (latitude2, longitude2))
    return distance

train["manhattandistance"] = train.apply(lambda row: distance(row["pickup_latitude"], 
                                               row["pickup_longitude"], 
                                               row["dropoff_latitude"], 
                                               row["dropoff_longitude"]), axis=1)

#Removing all records with a distance of zero
train = train[train.straightdistance>0]

#Showing a histogram of the distance
train[train.manhattandistance<25].manhattandistance.hist(bins=100)
plt.xlabel('Distance in KM')
plt.show()


# In[ ]:


#Identify airports

#JFK Airport
train['jfkpickup'] = train.apply(lambda row: haversine((40.6413, -73.7781), (row[4], row[3])), axis = 1)
train['jfkdropoff'] = train.apply(lambda row: haversine((row[2], row[1]), (40.6413, -73.7781)), axis = 1)

#La Guardia Airport
train['lgapickup'] = train.apply(lambda row: haversine((40.7769, -73.8740), (row[4], row[3])), axis = 1)
train['lgadropoff'] = train.apply(lambda row: haversine((row[2], row[1]), (40.7769, -73.8740)), axis = 1)

#Newark Airport
train['ewrpickup'] = train.apply(lambda row: haversine((40.6895, -74.1745), (row[4], row[3])), axis = 1)
train['ewrdropoff'] = train.apply(lambda row: haversine((row[2], row[1]), (40.6895, -74.1745)), axis = 1)

#Give a 1 to the rides that were picked up or dropped off near the airport
train['airportdropoff'] = np.where((train['jfkdropoff']<.5) | (train['lgadropoff']<.5) | (train['ewrdropoff']<.5), 1, 0)
train['airportpickup'] = np.where((train['jfkpickup']<.5) | (train['lgapickup']<.5) | (train['ewrpickup']<.5), 1, 0)


# In[ ]:


#Remove columns not longer needed
train = train.drop(['date','time','week_day','workingweek','pickup_longitude','pickup_latitude','dropoff_latitude','dropoff_longitude'], axis=1)

#Round columns
train = train.round({'straightdistance': 3, 'manhattandistance': 3, 'jfkpickup':3, 'lgapickup':3, 'ewrpickup':3, 'jfkdropoff':3, 'lgadropoff':3, 'ewrdropoff':3})

#Show final dataset
train.head()


# In[ ]:


#Separate data into input and output variables
data_labels = pd.DataFrame(train['fare_amount'].copy())
data_data = train.drop('fare_amount', axis=1)

#Separate into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(data_data, data_labels, test_size=0.30, random_state=1)


# In[ ]:


#Display training progress
class progress(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    print('Epoch:',epoch)

#define model
num_epochs = 200
batch_size = 10
learning_rate = .001

model = Sequential()
model.add(Dense(18, input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dense(12, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(6, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(3, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='linear'))
adam = optimizers.adam(lr=learning_rate)
model.compile(loss='mse', optimizer='adam')

model.summary()


# In[ ]:


#fit model
history = model.fit(X_train, y_train, epochs=num_epochs, batch_size = batch_size, verbose=0, callbacks=[progress()])


# In[ ]:


#Store predicted values
predicted_fare_amount = pd.DataFrame(model.predict(X_test))
predicted_fare_amount.columns = ['predicted_fare_amount']

#Create dataframe with predicted vs actual values
final = pd.merge(predicted_fare_amount, y_test, left_index=True, right_index=True)
final.to_csv('predictions.csv', sep=',')
final.head()

#Create dataframe with loss function
loss_data = pd.DataFrame.from_dict(history.history)
loss_data.to_csv('loss.csv', sep=',')


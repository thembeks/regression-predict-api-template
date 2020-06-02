"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Fetch training data and preprocess for modeling
train= pd.read_csv('https://raw.githubusercontent.com/thembeks/Regression-Sendy-Logistics-Challenge-Team-14/Predict/Train.csv')


#Drop columns not in test
copy=train.copy()
copy.drop(['Vehicle Type','Arrival at Destination - Day of Month','Arrival at Destination - Weekday (Mo = 1)','Arrival at Destination - Time', 'Order No'], axis=1, inplace=True)

#Remove strings in User ID and Ride Id
copy[['a','b', 'UserId No']] = copy["User Id"].str.split("_", expand = True)
copy[['c','d','RiderId No']] = copy["Rider Id"].str.split("_",  expand = True)

copy.drop(["User Id",'Rider Id', 'a', 'b','c','d'], axis=1, inplace = True)

copy.rename(columns={'UserId No':'User Id'}, inplace=True)
copy.rename(columns={'RiderId No':'Rider Id'}, inplace=True)

## convert time objects to datetime objects
copy['Placement - Time'] = pd.to_datetime(copy['Placement - Time'])
copy['Confirmation - Time'] = pd.to_datetime(copy['Confirmation - Time'])
copy['Arrival at Pickup - Time'] = pd.to_datetime(copy['Arrival at Pickup - Time'])
copy['Pickup - Time'] = pd.to_datetime(copy['Pickup - Time'])

### change time variables to difference in time in seconds to keep time stationary and appropriate for our model
copy['Time Placement to Confirmation'] = (copy['Confirmation - Time'] - copy['Placement - Time']).dt.seconds
copy['Time Confirmation to PickupArrival'] = (copy['Arrival at Pickup - Time'] - copy['Confirmation - Time']).dt.seconds
copy['Time Arrival to Pickup'] = (copy['Pickup - Time'] - copy['Arrival at Pickup - Time']).dt.seconds


### drop the old columns. Note we will keep one for analysis purposes 'Placement_Time'
#.drop(['Confirmation - Time', 'Arrival at Pickup - Time', "Pickup - Time"], axis= 1, inplace=True)



#Drop Columns
copy.drop(['Pickup - Time','Arrival at Pickup - Time','Confirmation - Time','Placement - Time', 'Rider Id','User Id'], axis=1, inplace=True)

#Rearrange dataframe
cols = list(copy.columns.values)
cols.pop(cols.index('Time from Pickup to Arrival')) 

copy = copy[cols+['Time from Pickup to Arrival']]

#Ecode categorical column
copy['Personal or Business'].unique()
Bdict = {'Personal': 0, 'Business': 1}
copy['Personal or Business'] = copy['Personal or Business'].map(Bdict)

#Replace NAN values
copy= copy.replace(np.nan, 0)

x = copy.drop(['Time from Pickup to Arrival'], axis=1)
y = copy['Time from Pickup to Arrival']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 50)



# Fit model


from sklearn.linear_model import LinearRegression
lm = LinearRegression(normalize=True)
lm.fit(X_train, y_train)

# Pickle model for use within our API

import pickle

model_save_path = "newer_model.pkl"
with open(model_save_path,'wb') as file:
    pickle.dump(lm,file)

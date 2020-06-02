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


df_rider = pd.read_csv(
    'https://raw.githubusercontent.com/thembeks/Regression-Sendy-Logistics-Challenge-Team-14/Predict/Riders.csv')

df_train = pd.read_csv(
    'https://raw.githubusercontent.com/thembeks/Regression-Sendy-Logistics-Challenge-Team-14/Predict/Train.csv')

df_clean=df_train.copy()
df_clean = df_train.merge(df_rider, how='left', left_on = 'Rider Id', right_on = 'Rider Id')

## convert time objects to datetime objects
df_clean['Placement - Time'] = pd.to_datetime(df_clean['Placement - Time'])
df_clean['Confirmation - Time'] = pd.to_datetime(df_clean['Confirmation - Time'])
df_clean['Arrival at Pickup - Time'] = pd.to_datetime(df_clean['Arrival at Pickup - Time'])
df_clean['Pickup - Time'] = pd.to_datetime(df_clean['Pickup - Time'])

### change time variables to difference in time in seconds to keep time stationary and appropriate for our model
df_clean['Time Placement to Confirmation'] = (df_clean['Confirmation - Time'] - df_clean['Placement - Time']).dt.seconds
df_clean['Time Confirmation to PickupArrival'] = (df_clean['Arrival at Pickup - Time'] - df_clean['Confirmation - Time']).dt.seconds
df_clean['Time Arrival to Pickup'] = (df_clean['Pickup - Time'] - df_clean['Arrival at Pickup - Time']).dt.seconds


### drop the old columns. Note we will keep one for analysis purposes 'Placement_Time'
df_clean.drop(['Confirmation - Time', 'Arrival at Pickup - Time', "Pickup - Time"], axis= 1, inplace=True)

## drop values that are different as they are the outliers
df_clean.drop(df_clean[df_clean['Placement - Day of Month'] != df_clean['Confirmation - Day of Month']].index, inplace = True)

## drop all other columns relating to the day of month and day of week
df_clean.drop(['Placement - Day of Month','Placement - Weekday (Mo = 1)', 'Confirmation - Day of Month', 'Confirmation - Weekday (Mo = 1)', 'Arrival at Pickup - Day of Month',
       'Arrival at Pickup - Weekday (Mo = 1)', 'Pickup - Day of Month', 'Pickup - Weekday (Mo = 1)'], axis = 1, inplace = True)

## We will drop the columns that are not in the test data set to keep consistency
df_clean.drop(['Arrival at Destination - Day of Month', 'Arrival at Destination - Weekday (Mo = 1)', 'Arrival at Destination - Time'], axis = 1, inplace = True)

## Note that for Placement Time column we just want the time and not date
df_clean['Placement - Time'] = df_clean['Placement - Time'].dt.time

## Note we need to convert platform type from numeric to categorical since its categorical
df_clean['Platform Type'] = df_clean['Platform Type'].map({3: 'plat 3', 1: 'plat 1', 2: 'plat 2', 4:'plat 4'})


#df_clean['Temperature'] = round(df_clean.groupby(['Day of Month'])['Temperature'].apply(lambda x: x.fillna(x.median())), 1)
df_clean['Precipitation in millimeters'].fillna(0.0, inplace = True)
df_clean['Temperature'].fillna(0.0, inplace = True)

df_clean.drop(['Order No', 'User Id', 'Rider Id', 'Placement - Time'], axis =1, inplace=True)
df_clean = pd.get_dummies(df_clean, drop_first = True)



#df_sig_test = df_clean.copy()

#df_sig_test.drop(['Platform_Type_plat_2', 'Platform_Type_plat_3', 'Platform_Type_plat_4','Personal_or_Business_Personal'], axis = 1, inplace = True)

df_train_clean = df_clean.copy()
q = df_train_clean['Time from Pickup to Arrival'].quantile(0.98)
data_1 = df_train_clean[(df_train_clean['Time from Pickup to Arrival']<q) & (df_train_clean['Time from Pickup to Arrival'] > 0)]
#sns.distplot(data_1['Time_from_Pickup_to_Arrival'])

p = df_train_clean['No_Of_Orders'].quantile(0.99)
data_2 = data_1[data_1['No_Of_Orders']<p]
#sns.distplot(data_2['No_Of_Orders'])

u = df_train_clean['Time Placement to Confirmation'].quantile(0.99)
data_3 = data_2[(data_2['Time Placement to Confirmation']<u) & (data_2['Time Placement to Confirmation']>0)]
df_no_outliers = data_3


cols = list(df_no_outliers.columns.values)
cols.pop(cols.index('Time from Pickup to Arrival')) 

df_no_outliers  = df_no_outliers [cols+['Time from Pickup to Arrival']]

model_features=['Distance (KM)', 'Temperature', 'Precipitation in millimeters',
       'Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long',
       'No_Of_Orders', 'Age', 'Average_Rating', 'No_of_Ratings',
       'Time Placement to Confirmation', 'Time Confirmation to PickupArrival',
       'Time Arrival to Pickup', 'Platform Type_plat 2',
       'Platform Type_plat 3', 'Platform Type_plat 4',
       'Personal or Business_Personal', 'Time from Pickup to Arrival']


x= df_no_outliers.iloc[:, :-1].values
y = df_no_outliers.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


reg = LinearRegression()
reg.fit(x_train, y_train)

import pickle

model_save_path = "new_model.pkl"
with open(model_save_path,'wb') as file:
    pickle.dump(reg,file)

"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])
    df_clean=feature_vector_df
    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    df_rider = pd.read_csv(
    'https://raw.githubusercontent.com/thembeks/Regression-Sendy-Logistics-Challenge-Team-14/Predict/Riders.csv')

    df_train = pd.read_csv(
    'https://raw.githubusercontent.com/thembeks/Regression-Sendy-Logistics-Challenge-Team-14/Predict/Train.csv')

    #df_clean=df_train.copy()
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

    df_no_outliers=df_no_outliers[model_features]
    return df_no_outliers

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()

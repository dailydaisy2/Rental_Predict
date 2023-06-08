import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pickle

data = pd.read_csv('mudah-apartment-kl-selangor.csv')

# data type conversion for monthly rent, rooms, size, location
# monthly_rent
# remove the 'RM', 'per month' and spaces from strings
data['monthly_rent'] = data['monthly_rent'].str.replace('RM', '').str.replace('per month', '').str.replace(' ', '')
# convert to numeric data type
data['monthly_rent'] = pd.to_numeric(data['monthly_rent'])

# rooms
# replace 'More than 10' with 10
data['rooms'] = data['rooms'].replace('More than 10', 10)
# convert to numeric data type
data['rooms'] = pd.to_numeric(data['rooms'])

# size
# remove the 'sq.ft.' string 
data['size'] = data['size'].str.replace(' sq.ft.', '')
# convert to numeric data type
data['size'] = pd.to_numeric(data['size'])

# location
# remove the 'KL' & 'Selangor' from location
data['location'] = data['location'].str.split('-').str[-1].str.strip()

data.info()

# to remove duplicated rows
data.drop_duplicates(inplace=True)
data.shape

# keep useful columns only
data = data[['monthly_rent', 'property_type', 'rooms', 'parking', 'size', 'furnished', 'region']]

# reduce the attribute of property_type
data.loc[data['property_type'].isin(['Bungalow House', 'Soho', 'Condo / Services residence / Penthouse / Townhouse', 'Residential', 'Houses']), 'property_type'] = 'Others'

# label encode categorical data to numeric
label_encoder = LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])

# fill in missing value with mean
for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())
print(data.isna().sum())

# select only 1000 rows to reduce pickled size
data2 = data[:1000]

# making data a numpy array like
x = data2.drop(['monthly_rent'], axis=1)
y = data2.monthly_rent
x = x.values
y = y.values

# split data into train 70% & test 30%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)

# standardzing the data
stds = StandardScaler()
scaler = stds.fit(x_train)
rescaledx = scaler.transform(x_train)

# selecting and fitting the model for training
model = RandomForestRegressor()
model.fit(rescaledx, y_train)
# saving the trained mode
pickle.dump(model, open('rf_model.pkl', 'wb'))
# saving StandardScaler
pickle.dump(stds, open('scaler.pkl', 'wb'))

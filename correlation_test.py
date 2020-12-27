#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:24:12 2020

@author: yojeremijenko-conel
"""
from imageai.Detection import ObjectDetection
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import timeit
import ast 
from scipy import stats
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#%%
os.chdir('/Users/yojeremijenko-conel/Documents/GitHub/Satelite-Research-Project/FirstTraining')

data = pd.read_csv('df_detections.csv')

#%%
objects = []

def lit(s):
    return ast.literal_eval(s)

data['pic_2_2_detections'] = data['pic_2_2_detections'].apply(lit)
data['pic_1_detections'] = data['pic_1_detections'].apply(lit)
data['pic_2_1_detections'] = data['pic_2_1_detections'].apply(lit)
data['pic_2_3_detections'] = data['pic_2_3_detections'].apply(lit)
data['pic_2_4_detections'] = data['pic_2_4_detections'].apply(lit)
data['pic_2_5_detections'] = data['pic_2_5_detections'].apply(lit)





objects = []

for i in data.pic_2_2_detections:
    for j in i:
        objects.append(j['name'])

for i in data.pic_1_detections:
    for j in i:
        objects.append(j['name'])

for i in data.pic_2_1_detections:
    for j in i:
        objects.append(j['name'])

for i in data.pic_2_3_detections:
    for j in i:
        objects.append(j['name'])

for i in data.pic_2_4_detections:
    for j in i:
        objects.append(j['name'])

for i in data.pic_2_5_detections:
    for j in i:
        objects.append(j['name'])

objects = set(objects)

#%%

for i in objects:
    lst = []
    for j in range(len(data.pic_2_2_detections)):
        count = 0
        for k in data.pic_2_2_detections[j]:
            if i == k['name']:
                count += 1
        for k in data.pic_1_detections[j]:
            if i == k['name']:
                count += 1
        for k in data.pic_2_1_detections[j]:
            if i == k['name']:
                count += 1
        for k in data.pic_2_3_detections[j]:
            if i == k['name']:
                count += 1
        for k in data.pic_2_4_detections[j]:
            if i == k['name']:
                count += 1
        for k in data.pic_2_5_detections[j]:
            if i == k['name']:
                count += 1
        lst.append(count)
    data[i] = lst

#%%

for i in objects:
    if np.count_nonzero(data[i]) == 1 and max(data[i]) == 1:
        data = data.drop(i, axis=1)
#%%    
data = data.set_index(['Unnamed: 0', 'Unnamed: 1'])
#%%
path = "Income_SDGs_MPI_Oct2019.csv"
df = pd.read_csv(path, encoding = "ISO-8859-1")
df = df[['pic_1', 'gpsLatitude','gpsLongitude', 'address','total_income_2_b', 'total_deprivations', 'Mobile_Phone', 'Bicyle', 'Motorbike_Motorcycle',
       'Animal_Traction_Cart', 'Car_Truck', 'Motor_Boat', 'Wealth_Index',
       'total_deprivations_2', 'Poor_MPI', 'Severely_Poor_MPI']]
df = df[df['pic_1'].isin(data.pic_1)]

data = pd.concat([data, df], axis=1, sort=False)

#%%
data.to_csv('objects.csv')
#%%
car_truck = (data.car + data.truck) > 0
no_car_truck = car_truck == False
print(np.mean(data.Car_Truck[car_truck]))
print(np.mean(data.Car_Truck[no_car_truck]))

#hyothesis test to determine the predictive power of car and truck detections on survey reports
observed = np.mean(data.Car_Truck[car_truck])
sims = []
for i in range(10000):
    sample = np.random.choice([True, False], p = [np.mean(car_truck), 1 - np.mean(car_truck)], size = 1200)
    sims.append(np.mean(data.Car_Truck[sample]))


#%%    
#pval to reject the null: houses where cars or trucks are detected are no more likely than those without to report ownership
#of a car or truck 
pval = (np.count_nonzero(np.array(sims) >= observed)/10000) * 100
print(pval)
#pval of 0 given for a sample of 10,000 simulations under the null, thus we can reject the null hypothesis


#%%
#same test for motercycle 

motor = (data.motorcycle) > 0
no_motor = motor == False
print(np.mean(data.Motorbike_Motorcycle[motor]))
print(np.mean(data.Motorbike_Motorcycle[no_motor]))

#%%
#hyothesis test to determine the predictive power of car and truck detections on survey reports
observed_m = np.mean(data.Motorbike_Motorcycle[motor])

sims = []
for i in range(10000):
    sample = np.random.choice([True, False], p = [np.mean(motor), 1 - np.mean(motor)], size = 1200)
    sims.append(np.mean(data.Motorbike_Motorcycle[sample]))

#%%
#pval to reject null for motercycle
pval_m = (np.count_nonzero(np.array(sims) >= observed_m)/10000) * 100
print(pval_m)

#pval of 44.2% for the null, thus we cannot reject the null for motercycles

#%%
#logistic regression for cars and trucks
logres = data[['car','truck', 'Car_Truck']]
logres.index = range(1200)
#plt.scatter(car_truck, data.Car_Truck)
#%%

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(logres[['car', 'truck']],logres.Car_Truck,train_size=0.75)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

model.predict_proba(X_test)

print(model.score(X_test,y_test))
print(np.mean(y_predicted))

#%%
slope, intercept, r_value, p_value, std_err = stats.linregress(data.car, data.total_deprivations_2)

print('Slope: ',slope,'\nIntercept: ',intercept , '\nr2', r_value**2)

x = data.car
y = data.total_deprivations_2
def predict_y_for(x):
    return slope * x + intercept

plt.scatter(x,y)
plt.plot(x, predict_y_for(x), c='r')
plt.show()

#%%
#logistic regression for cars and trucks
#logres = data[['train', 'potted plant','car', 'truck', 'person', 
#              'Severely_Poor_MPI', 'Poor_MPI', 'total_deprivations_2', 'Wealth_Index',
#               'total_income_2_b', 'total_deprivations']]

logres = data[['chair', 'truck', 'bottle', 'bird', 'fire hydrant', 'car', 'umbrella',
       'person', 'bus', 'backpack', 'bench', 'handbag', 'potted plant',
       'motorcycle', 'bowl', 'train', 'dog', 'Severely_Poor_MPI', 'Poor_MPI',
       'total_deprivations_2', 'Wealth_Index', 'total_income_2_b',
       'total_deprivations']]
logres['Severely_Poor_MPI'] = logres['Severely_Poor_MPI'] == 'Poor'
logres['Poor_MPI'] = logres.Poor_MPI == 'Poor'

mean_counts = logres.applymap(lambda x: x == 0).mean()

logres.index = range(1200)

#%%
#more expoloratory data analysis
import seaborn as sn

corrMatrix = logres.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()
type(corrMatrix)
corrs = corrMatrix[['Severely_Poor_MPI', 'Poor_MPI', 'total_deprivations_2', 'Wealth_Index',
               'total_income_2_b', 'total_deprivations']]
corrs = corrs.loc[['chair', 'truck', 'bottle', 'bird', 'fire hydrant', 'car', 'umbrella',
       'person', 'bus', 'backpack', 'bench', 'handbag', 'potted plant',
       'motorcycle', 'bowl', 'train', 'dog']]
corrs_abs = corrs.applymap(abs)    
corrs_abs_t = corrs_abs.transpose()

sn.heatmap(corrs, annot=True)
plt.show()

wealth_corrs = corrs_abs.Wealth_Index
wealth_corrs = wealth_corrs[wealth_corrs >= np.mean(wealth_corrs)]
indx = wealth_corrs.index
wealth_corrs = corrs.Wealth_Index[indx]

#plt.scatter(car_truck, data.Car_Truck)
#%%

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X_train, X_test, y_train, y_test = train_test_split(logres[['train', 'truck', 'car', 'person']],logres.total_deprivations_2,train_size=0.8)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

model.predict_proba(X_test)

print(model.score(X_test,y_test))
print(np.mean(y_predicted))
print(np.mean(logres.total_deprivations_2))
#%% 
#logistric regression function

def logreg(col):
    indep = list(corrs_abs[col].index[corrs_abs[col] >= np.mean(corrs_abs[col])])
    print('Regressors: ', indep)
    X_train, X_test, y_train, y_test = train_test_split(logres[indep],logres[col],train_size=0.8)
    model= LogisticRegression()
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    print("Model Score: ", model.score(X_test,y_test))
    print('Mean Predicted: ', np.mean(y_predicted))
    print('Mean of all observations: ', np.mean(logres[col]))
    print("Number of categories: ", len(set(logres[col])))
    print('SDG CLASSIFIER \n')
    clf = SGDClassifier(loss="hinge", penalty="l2")
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    print('Score: ', clf.score(X_test,y_test))


#%%
data = pd.read_csv('objects.csv')
rgb = pd.read_csv('rgb_mean.csv')
data['r_mean'] = rgb['r_mean']
data['g_mean'] = rgb['g_mean']
data['b_mean'] = rgb['b_mean']


x = data[['truck', 'car', 'person', 'potted plant', 'bowl', 'r_mean', 'g_mean', 'b_mean']]
y = data['Wealth_Index']
linear_regression = LinearRegression()
linear_regression.fit(x, y)

pred = linear_regression.predict(x)


print('Coefficients: \n', linear_regression.coef_)

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y, pred))

print('R2: %.4f'
      % r2_score(y, pred))





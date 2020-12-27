#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:11:42 2020

@author: yojeremijenko-conel
"""

import sys
import skimage.io
import skimage.viewer
import numpy as np
import os
from matplotlib import pyplot as plt
#import opencv

# read original image, in full color, based on command
# line argument
image = skimage.io.imread(fname=os.path.join(os.getcwd(), 'flattened_pics', '1548779251774.jpg'))

# display the image
#image = os.path.join(os.getcwd(), 'flattened_pics', '1548779251774.jpg')
#viewer = skimage.viewer.Viewer(image)
#viewer.show()

colors = ("r", "g", "b")
channel_ids = (0, 1, 2)
hist_dict = dict()
# create the histogram plot, with three lines, one for
# each color
plt.xlim([0, 256])
for channel_id, c in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        image[:, :, channel_id], bins=2, range=(0, 256)
    )
    hist_dict[c] = histogram
    plt.plot(bin_edges[0:-1], histogram, color=c)
    

plt.xlabel("Color value")
plt.ylabel("Pixels")

plt.show()
#%%
def color_dist(x):
    image = skimage.io.imread(fname=os.path.join(os.getcwd(), 'flattened_pics', x))
    colors = ("r", "g", "b")
    channel_ids = (0, 1, 2)
    hist_dict = dict()
    # create the histogram plot, with three lines, one for
    # each color
    plt.xlim([0, 256])
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(
            image[:, :, channel_id], bins=10, range=(0, 256)
            )
        hist_dict[c] = histogram
    return hist_dict
    
#%%
import pandas as pd
df = pd.read_csv('objects.csv')
pics = ['pic_1', 'pic_2_1','pic_2_2',  'pic_2_3',  'pic_2_4', 'pic_2_5']

for i in pics:
    p = i + 'rgb'
    df[p] = df[i].apply(color_dist)
    print(df[p][0])
#%%
rgb = df[['Unnamed: 0', 'Unnamed: 1', 'municipality', 'commune', 'pic_1',
        'pic_2_1', 'pic_2_2', 
       'pic_2_3',  'pic_2_4',  'pic_2_5', 'pic_1.1','gpsLatitude', 'gpsLongitude',
       'address', 'total_income_2_b', 'total_deprivations',  'Wealth_Index',
       'total_deprivations_2', 'Poor_MPI', 'Severely_Poor_MPI', 'pic_1rgb',
       'pic_2_1rgb', 'pic_2_2rgb', 'pic_2_3rgb', 'pic_2_4rgb', 'pic_2_5rgb']]


r_sum = lambda x: np.sum(x['r'])
g_sum = lambda x: np.sum(x['g'])
b_sum = lambda x: np.sum(x['b'])


pics = ['pic_1', 'pic_2_1','pic_2_2',  'pic_2_3',  'pic_2_4', 'pic_2_5']

for i in pics:
    p = i + 'rgb'
    r = 'red_' + i
    rgb[r] = rgb[p].apply(r_sum)
    g = 'green_' + i
    rgb[g] = rgb[p].apply(g_sum)
    b = 'blue_' + i
    rgb[b] = rgb[p].apply(b_sum)

rgb.to_csv('rbg_dist.csv')

#%%
    
print(np.array(image).shape)
tuple(image.mean(axis=0))[0]


def getAverageRGBN(x):
  """
  Given PIL Image, return average value of color as (r, g, b)
  """
  image = skimage.io.imread(fname=os.path.join(os.getcwd(), 'flattened_pics', x))
  return np.array(image).mean(axis=(0,1))


for i in pics:
    p = i + 'rgb'
    df[p] = df[i].apply(getAverageRGBN)
    print(df[p][0])
#%%
rgb = df[['Unnamed: 0', 'Unnamed: 1', 'municipality', 'commune', 'pic_1',
        'pic_2_1', 'pic_2_2', 
       'pic_2_3',  'pic_2_4',  'pic_2_5', 'pic_1.1','gpsLatitude', 'gpsLongitude',
       'address', 'total_income_2_b', 'total_deprivations',  'Wealth_Index',
       'total_deprivations_2', 'Poor_MPI', 'Severely_Poor_MPI', 'pic_1rgb',
       'pic_2_1rgb', 'pic_2_2rgb', 'pic_2_3rgb', 'pic_2_4rgb', 'pic_2_5rgb']]


for i in pics:
    p = i + 'rgb'
    r = 'red_' + i
    rgb[r] = rgb[p].apply(lambda x: x[0])
    g = 'green_' + i
    rgb[g] = rgb[p].apply(lambda x: x[1])
    b = 'blue_' + i
    rgb[b] = rgb[p].apply(lambda x: x[2])


rgb.to_csv('rgb_mean.csv')

#%%
import seaborn as sn
c1 = rgb[['total_income_2_b',
       'total_deprivations', 'Wealth_Index', 'total_deprivations_2',
       'Poor_MPI', 'Severely_Poor_MPI', 'red_pic_1', 'green_pic_1',
       'blue_pic_1']]


corrMatrix = c1.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()
#%%
c2 = rgb[['total_income_2_b',
       'total_deprivations', 'Wealth_Index', 'total_deprivations_2',
       'Poor_MPI', 'Severely_Poor_MPI', 'red_pic_2_1', 'green_pic_2_1',
       'blue_pic_2_1']]


corrMatrix = c2.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()
#%%
c3 = rgb[['total_income_2_b',
       'total_deprivations', 'Wealth_Index', 'total_deprivations_2',
       'Poor_MPI', 'Severely_Poor_MPI', 'red_pic_2_2', 'green_pic_2_2',
       'blue_pic_2_2']]


corrMatrix = c3.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()
#%%
c4 = rgb[['total_income_2_b',
       'total_deprivations', 'Wealth_Index', 'total_deprivations_2',
       'Poor_MPI', 'Severely_Poor_MPI', 'red_pic_2_3', 'green_pic_2_3',
       'blue_pic_2_3']]


corrMatrix = c4.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

#%%
c5 = rgb[['total_income_2_b',
       'total_deprivations', 'Wealth_Index', 'total_deprivations_2',
       'Poor_MPI', 'Severely_Poor_MPI', 'red_pic_2_4', 'green_pic_2_4',
       'blue_pic_2_4']]


corrMatrix = c5.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

#%%

c6 = rgb[['total_income_2_b',
       'total_deprivations', 'Wealth_Index', 'total_deprivations_2',
       'Poor_MPI', 'Severely_Poor_MPI', 'red_pic_2_5', 'green_pic_2_5',
       'blue_pic_2_5']]


corrMatrix = c6.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

#%%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy import stats



slope, intercept, r_value, p_value, std_err = stats.linregress(rgb[['red_pic_1', 'green_pic_1',
       'blue_pic_1', 'red_pic_2_1', 'green_pic_2_1', 'blue_pic_2_1',
       'red_pic_2_2', 'green_pic_2_2', 'blue_pic_2_2', 'red_pic_2_3',
       'green_pic_2_3', 'blue_pic_2_3', 'red_pic_2_4', 'green_pic_2_4',
       'blue_pic_2_4', 'red_pic_2_5', 'green_pic_2_5', 'blue_pic_2_5']], rgb[['Wealth_Index']])

print('Slope: ',slope,'\nIntercept: ',intercept , '\nr2', r_value**2)
#%%
x = rgb[['red_pic_1', 'green_pic_1',
       'blue_pic_1', 'red_pic_2_1', 'green_pic_2_1', 'blue_pic_2_1',
       'red_pic_2_2', 'green_pic_2_2', 'blue_pic_2_2', 'red_pic_2_3',
       'green_pic_2_3', 'blue_pic_2_3', 'red_pic_2_4', 'green_pic_2_4',
       'blue_pic_2_4', 'red_pic_2_5', 'green_pic_2_5', 'blue_pic_2_5']]
y = rgb['Wealth_Index']
linear_regression = LinearRegression()
linear_regression.fit(x, y)

pred = linear_regression.predict(x)


print('Coefficients: \n', linear_regression.coef_)

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y, pred))

print('R2: %.4f'
      % r2_score(y, pred))
#%%
rgb['r_mean'] = rgb['red_pic_1'] + rgb['red_pic_2_1'] + rgb['red_pic_2_2'] + rgb['red_pic_2_3'] + rgb['red_pic_2_4'] + rgb['red_pic_2_5']
rgb['r_mean'] = rgb['r_mean']/6

rgb['g_mean'] = rgb['green_pic_1'] + rgb['green_pic_2_1'] + rgb['green_pic_2_2'] + rgb['green_pic_2_3'] + rgb['green_pic_2_4'] + rgb['green_pic_2_5']
rgb['g_mean'] = rgb['g_mean']/6

rgb['b_mean'] = rgb['blue_pic_1'] + rgb['blue_pic_2_1'] + rgb['blue_pic_2_2'] + rgb['blue_pic_2_3'] + rgb['blue_pic_2_4'] + rgb['blue_pic_2_5']
rgb['b_mean'] = rgb['b_mean']/6

#%%
mean_corr = rgb[['total_income_2_b',
       'total_deprivations', 'Wealth_Index', 'total_deprivations_2',
       'Poor_MPI', 'Severely_Poor_MPI','r_mean', 'g_mean', 'b_mean' ]]


corrMatrix = mean_corr.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()
#%%

x = rgb[['r_mean', 'g_mean', 'b_mean']]
y = rgb['Wealth_Index']
linear_regression = LinearRegression()
linear_regression.fit(x, y)

pred = linear_regression.predict(x)


print('Coefficients: \n', linear_regression.coef_)

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y, pred))

print('R2: %.4f'
      % r2_score(y, pred))





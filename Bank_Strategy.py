#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:43:31 2020

@author: Vineeth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels.api as sm

raw_data = pd.read_csv('/Users/Vineeth/Documents/Studies/The Data Science Course 2019 - All Resources/Part_5_Advanced_Statistical_Methods_(Machine_Learning)/S36_L247/Bank-data.csv')

data=raw_data.copy()
data = data.drop(['Unnamed: 0'],axis=1)
data['y'] = data['y'].map({'yes':1,'no':0})

variables = ['interest_rate', 'credit', 'march', 'may', 'previous', 'duration']

y = data['y']
x1 = data['duration']

x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
result_log = reg_log.fit()

result_log.summary()

plt.scatter(x1,y,color='red')
plt.xlabel('Duration',fontsize=20)
plt.ylabel('Approved',fontsize=20)
plt.show()

x1_all = data[variables]

x_all = sm.add_constant(x1_all)
reg_all = sm.Logit(y,x_all)
result_all = reg_all.fit()

result_all.summary()

def confusion_matrix(x,y,model):
    predicted_values = model.predict(x)
    bins = np.array([0,0.5,1])
    cm = np.histogram2d(y,predicted_values,bins)[0]
    accuracy = (cm[0,0]+cm[1,1])/cm.sum()
    return cm, accuracy
    
    
confusion_matrix(x,y,result_log)

confusion_matrix(x_all,y,result_all)

#Testing

test_data = pd.read_csv('/Users/Vineeth/Documents/Studies/The Data Science Course 2019 - All Resources/Part_5_Advanced_Statistical_Methods_(Machine_Learning)/S36_L250/Bank-data-testing.csv')

test_data = test_data.drop(['Unnamed: 0'],axis=1)
test_data['y'] = test_data['y'].map({'yes':1,'no':0})
test_data.describe()

test_y = test_data['y']
test_x1 = test_data[variables]
test_x = sm.add_constant(test_x1)

confusion_matrix(test_x,test_y,result_all)

confusion_matrix(x_all,y,result_all)





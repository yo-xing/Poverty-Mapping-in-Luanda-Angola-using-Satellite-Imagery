#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 14:03:25 2020

@author: yojeremijenko-conel
"""

#%%
#2

def func_two(x):
    if x%3 != 0 and x%2 == 0:
        return True
    elif x%3 == 0 and x%2 != 0:
        return True
    else:
        return False
    
#%%
        
def func_three():
    statement = input('What statement would you like to classify:')
    expert = input('Is that statement True, False, or Unknown according to Experts or most logical interpretation of the evidence? Choose only: True, False or Unknown:')
    donald = input('Is that statement True, False, or Unknown according to Donald Trump? Choose only: True, False or Unknown:')
    if expert == 'True':
        if donald == 'True': 
            c = 'True'
        elif donald == 'False':
            c = 'Alternative Falsehood'
        elif donald == 'Unknown':
            c = 'Alternative Unknown'
    elif expert == 'Unknown':
        if donald == 'True': 
            c = 'Alternative Fact'
        elif donald == 'False':
            c = 'Alternative Falsehood'
        elif donald == 'Unknown':
            c = 'Unknown'
    elif expert == 'False':
        if donald == 'True': 
            c = 'Alternative Fact'
        elif donald == 'False':
            c = 'False'
        elif donald == 'Unknown':
            c = 'Alternative Unknown'
    print('The classification of your question: ' + statement + ' is ' + c)
        
        
        
 #%%
import numpy as np

def guess_my_number(num = 50, i = 0, prev = 0):
        
    if i == 3:
        if num > prev:
            guess = np.random.choice(range(prev, num))
            print('My guess for your number is: ' + str(guess))
            
        else:
            guess = np.random.choice(range(num, prev))
            print('My guess for your number is: ' + str(guess)) 
        return guess
    if i == 0:
        print('Please think of a number between 0 and 100')
    q = input('is your number greater than or equal to '+ str(num) + '? Choose only: yes, no ')
    if q == 'yes':
        if prev > num:
            temp = round((num + prev)/2)
        else:
            temp = round((num + 100)/2)
    else:
        if prev < num:
            temp = round((num + prev)/2)
        else:
            temp = round((num + 0)/2)
        
    prev = num
    num = temp
    guess_my_number(num , i + 1, prev)
        
        
            
            
            
            
            
            
            
            
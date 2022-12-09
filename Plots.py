#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 10:37:45 2022

@author: sjoerdvanvelzen
"""

import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

df = pd.read_excel(r'/Users/sjoerdvanvelzen/Documents/Econometrie/Master/Blok 2/Computer Science for Business Analytics/Assignment/Results final.xlsx')

PQ = df['PQ']
Precision = df['Precision']

PC = df['PC']
Recall = df['Recall']

FracComp = df['Frac of comp']

F1Star = df['F1*']
F1 = df['F1']


#plotPQ = plt.plot(FracComp, PQ)
#plt.ylabel('Pair Quality')
#plt.xlabel('Fraction of comparisons')

# plotPC = plt.plot(FracComp, PC)
# plt.ylabel('Pair Completeness')
# plt.xlabel('Fraction of comparisons')

#plotF1 = plt.plot(FracComp, F1)
#plt.ylim([0, 0.4])
#plt.ylabel('F1-measure') 
#plt.xlabel('Fraction of comparisons')

plotF1star = plt.plot(FracComp, F1Star)
plt.ylabel('F1*-measure')
plt.xlabel('Fraction of comparisons')
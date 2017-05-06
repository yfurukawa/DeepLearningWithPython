'''
Created on 2016/11/20

@author: furu
'''
import numpy as np

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a -c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

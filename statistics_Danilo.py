#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:06:04 2021

@author: Danilo
"""
# 
import numpy as np
from scipy import stats
# ----------

# ----------
def di_acc(obs, olam):
    '''
    Calculate difference (di) between observation and simulation for total acc. prec.
    '''
    diff =  olam - obs          
    return diff
# ----------
def BIAS(obs, olam):
    '''
    Calculate model Bias, or sistematic error based on the 'di' 
    '''
    diff = di_acc(obs, olam)
    tmp = diff.sum('lon').sum('lat').values
    return tmp/(diff.shape[0]*diff.shape[1])
# ----------
def MAE(obs, olam):
    '''
    Calculate model Mean Absolute Error based on the 'di' 
    '''    
    abs_diff = abs(di_acc(obs, olam))
    tmp = abs_diff.sum('lon').sum('lat').values
    return tmp/(abs_diff.shape[0]*abs_diff.shape[1])    
# ----------
def MSE(obs, olam):
    '''
    Calculate model Mean Square Error based on the 'di' 
    '''     
    diff_sqr = di_acc(obs, olam)**2
    tmp = diff_sqr.sum('lon').sum('lat').values
    N = diff_sqr.size
    return tmp/N
# ----------
def MSE_diss(obs, olam):
    '''
    Calculate the model Dissipative Mean Square Error based on the 'di' 
    ''' 
    tmp1 = (np.nanstd(obs.values) - np.nanstd(olam.values))**2
    tmp2 = (np.nanmean(obs.values) - np.nanmean(olam.values))**2
    return tmp1+tmp2
# ----------
def covariance(obs, olam):
    '''
    Calculate Covariance between model and reanalysis 
    '''   
    ave_olam = np.mean(olam)
    ave_obs = np.mean(obs)
    tmp = (olam - ave_olam) * (obs - ave_obs)
    tmp = tmp.sum('lon').sum('lat').values
    N = olam.size
    return tmp/(N-1)
# ----------
def Scorr(obs, olam):
    '''
    Calculate Spatial Correlation between model and reanalysis using 3 methods
    '''      
    cov = covariance(obs, olam)
    pcorr = cov/(np.nanstd(obs.values) * np.nanstd(olam.values))
    scorr, _ = stats.spearmanr(obs.values, olam.values,axis=None)
    kcorr, _ = stats.kendalltau(obs.values, olam.values)    
    return  pcorr, scorr, kcorr 

# ----------
def MSE_disp(obs, olam):
    '''
    Calculate the model Dispersive Mean Square Error based on the 'di' 
    '''  
    p = Scorr(obs, olam)[0]
#    p = Scorr(obs, olam)
    tmp1 = 2*(1-p)
    tmp2 = np.nanstd(obs.values) * np.nanstd(olam.values)
    return tmp1*tmp2
# ----------
def RMSE(obs, olam):
    '''
    Calculate model Root Mean Square Error based on the 'di' 
    '''        
    diff_sqr = di_acc(obs, olam)**2
    tmp = diff_sqr.sum('lon').sum('lat').values
    N = diff_sqr.size
    tmp = tmp/N
    return np.sqrt(tmp)
# ----------
def RMSE_bias(obs, olam):
    '''
    Calculate model RMSE after constant bias removal 
    '''      
    obs_mean = np.mean(obs)
    olam_mean = np.mean(olam)
    N = olam.size
    tmp = ((obs-obs_mean) - (olam-olam_mean))**2
    tmp = (tmp.sum('lon').sum('lat').values)/N
    return np.sqrt(tmp)
# ----------
def S_sqr(obs, olam):
    '''
    Calculate the Differences Variance (Sd^2)
    '''   
    diff = di_acc(obs, olam)
    N = diff.size
    diff_m = np.nanmean(diff.values)
    tmp = (diff-diff_m)**2
    tmp = tmp.sum('lon').sum('lat').values
    return tmp/(N-1)
# ----------
def ConcordanceIndex(obs, olam):
    '''
    Calculate Concordnce Index (Willmot, 1982) 
    '''   
    diff_sqr = di_acc(obs, olam)**2
    top = diff_sqr.sum('lon').sum('lat').values
    
    obs_m = np.mean(obs).values    
    tmp1 = np.abs(olam - obs_m)
    tmp2  = np.abs(obs - obs_m)
    tmp = (tmp1+tmp2)**2
    bottom = tmp.sum('lon').sum('lat').values
    
    return 1-(top/bottom)
# ----------
def D_pielke(obs, olam):
    '''
    Calculate Pielke Model's Dexterity  (Pielke, 2002)
    '''   
    obs_v = obs.values
    tmp1 = np.abs(1-(np.nanstd(olam.values)/np.nanstd(obs_v)))
    tmp2 = RMSE(obs, olam)/np.nanstd(obs_v)
    tmp3 = RMSE_bias(obs, olam)/np.nanstd(obs_v)
    return tmp1+tmp2+tmp3


def linear_regression(x, y):     
    N = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    
    B1_num = ((x - x_mean) * (y - y_mean)).sum()
    B1_den = ((x - x_mean)**2).sum()
    B1 = B1_num / B1_den
    
    B0 = y_mean - (B1*x_mean)
    
    reg_line = 'y = {} + {}β'.format(B0, round(B1, 3))
    
    return (B0, B1, reg_line)
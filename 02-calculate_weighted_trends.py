#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 10:00:51 2023

@author: Antti-Ilari Partanen (antti-ilari.partanen@fmi.fi)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm, gaussian_kde

'''

IPCC AR6 TS.3.2 Climate Sensitivity and Earth System Feedbacks:
"Based on process understanding, warming over the instrumental
record, and emergent constraints, the best estimate of TCR is 1.8°C,
the likely range is 1.4°C to 2.2°C and the very likely range is 1.2°C to
2.4°C. There is a high level of agreement among the different lines of
evidence (Figure TS.16c) (high confidence). {7.5.5}""

'''


datadir=Path('data')
inputdir=datadir / 'input'
outputdir=datadir / 'output'
figdir=Path('figures')



#open CMIP6 data
df_cmip6=pd.read_csv(outputdir / 'cmip6_raw_trends.csv', index_col=0)

def add_str(string_to_add, list_of_strings_in):
    '''
    Adds a string to each string of a list

    Parameters
    ----------
    string_to_add : str
        String to be appended at the end of each string in list_of_strings_in.
    list_of_strings_in : list
        A list of strings.

    Returns
    -------
    list_of_strings_out : list
        List of strings with string_to_add appended at the end of each string.

    '''
    list_of_strings_out=list_of_strings_in.copy()
    for i,string in enumerate(list_of_strings_out):
        list_of_strings_out[i]=string+string_to_add
    return list_of_strings_out

# Re-Calculate warming rates 
ssps3=['ssp119','ssp126', 'ssp585']
ssps2=['ssp126', 'ssp585']
obs_warming_trend=0.486 # K / decade
obs_heat_trend=0.56 # K / decade

# Calculate scaled trends for SSPs
df_cmip6[add_str('-scaled',ssps3)]=df_cmip6[ssps3].div(df_cmip6['historical'], axis=0)*obs_warming_trend


# Calculated heat index trends
df_cmip6[add_str('-heat_index_scaled',ssps3)]=df_cmip6[ssps3].div(df_cmip6['historical'], axis=0)*obs_heat_trend


#Open TCR data
data_tcr=pd.read_excel(inputdir / 'TCR.xlsx', index_col=0, sheet_name='TCR', skiprows=0)
df_cmip6['TCR']=data_tcr['TCR'].loc[df_cmip6.index]

#Remove models without TCR value
df_cmip6.dropna(subset=['TCR'], inplace=True)


#Formulate TCR distribution based on the IPCC report
NINETY_TO_ONESIGMA = norm.ppf(0.95)
tcr_norm=norm(loc=1.8, scale=0.6 / NINETY_TO_ONESIGMA)

# Calculate weights based on TCR
df_cmip6['weight']=tcr_norm.pdf(df_cmip6['TCR'])
df_cmip6['weight']=df_cmip6['weight']/df_cmip6['weight'].sum()



def calculate_means(df_in,ssps):
    df_out=df_in.copy()
    scens='historical|TCR|weight'
    for ssp in ssps:
        scens=scens+'|'+ssp
    #Select only certain ssps and drop rows with NaN values    
    df_out=df_out.filter(regex=scens).dropna(axis=0)
    # ssps_scaled=['historical']+ssps+add_str('-scaled', ssps)
    # df_out=df_cmip6[ssps_scaled+['TCR','weight']]
    #Normalize weights so they sum to 1
    df_out.loc[:,'weight']=df_out['weight']/df_out['weight'].sum()

    df_out.loc['posterior_mean']=df_out.iloc[:,0:-1].mul(df_out['weight'], axis=0).sum(axis=0)
    df_out.loc['prior_mean']=df_out.mean(axis=0)
    
    return df_out

#Calculate posterior trends from models with data for all SSPs and TCR
df_all_ssps=calculate_means(df_cmip6, ssps3)


# Calculate means for SSP1-2.6 and SSP5-8.5
df_all_models=calculate_means(df_cmip6, ssps2)


def calculate_kdes(df):
    columns=df.columns[0:-1] #Select all except weights
    kde_posterior=dict()
    kde_prior=dict()
    for column in columns:
        kde_prior[column]=gaussian_kde(df[column][0:-2])
        kde_posterior[column]=gaussian_kde(df[column][0:-2],weights=df['weight'][0:-2])
    return kde_prior, kde_posterior

kde_prior_all_ssps, kde_posterior_all_ssps=calculate_kdes(df_all_ssps)
kde_prior_all_models, kde_posterior_all_models=calculate_kdes(df_all_models)


# Calculate credible intervals for the prior and posterior distributions
def calculate_intervals(df_in,ssps,kde, kde_posterior):
    df_out=df_in.copy()
    intervals=dict()
    intervals_posterior=dict()
    
    scens=df_out.columns[:-1]
    
    for scen in scens:
        n=1000000
        seed=1
        intervals[scen]=np.quantile(kde[scen].resample(n,seed=seed),q=[0.025,0.5, 0.975])
        intervals_posterior[scen]=np.quantile(kde_posterior[scen].resample(n,seed=seed),q=[0.025,0.5, 0.975])
        df_out.loc['ci95_prior_low', scen]=intervals[scen][0]
        df_out.loc['ci95_prior_high', scen]=intervals[scen][2]
        df_out.loc['prior_median', scen]=intervals[scen][1]
        df_out.loc['ci95_posterior_low', scen]=intervals_posterior[scen][0]
        df_out.loc['ci95_posterior_high', scen]=intervals_posterior[scen][2]
        df_out.loc['posterior_median', scen]=intervals_posterior[scen][1]
    
    return intervals, intervals_posterior, df_out


intervals_all_ssps, intervals_posterior_all_ssps, df_all_ssps=calculate_intervals(df_all_ssps,ssps3,kde_prior_all_ssps, kde_posterior_all_ssps)
intervals_all_models, intervals_posterior_all_models, df_all_models=calculate_intervals(df_all_models,ssps2,kde_prior_all_models, kde_posterior_all_models)

def visualize(df, intervals, intervals_posterior, title='', figname=None, cut_warming_rate=False):
    
    #Do not plot intervals, means and medians along model data
    models=df.index[0:-8]
    
    fig,ax=plt.subplots(1,1, figsize=(7.5,6))
    
    #Select columns excluding TCR and weight
    xticklabels=df.columns[0:-2].sort_values()
    

    for i,column in enumerate(xticklabels):
        
        if i==len(xticklabels)-1:
            label_ci='95% credible interval (prior)'
            label_ci_posterior='95% credible interval (posterior)'
            label_scatter='Individual models'
            label_posterior_mean='Posterior mean'
            label_mean='Prior mean'
        else:
            label_ci='_no_legend'
            label_ci_posterior='_no_legend'
            label_scatter='_no_legend'
            label_posterior_mean='_no_legend'
            label_mean='_no_legend'
        
        #Plot individula models
        sc=ax.scatter(i*np.ones(len(df.loc[models, column])),
                      df.loc[models, column],c=df.loc[models,'weight'],
                      cmap='GnBu',edgecolors='black', label=label_scatter)
        
        
        
        #Plot parameters on how far away from the model scatter line draw
        # CIs and mean values
        prior_inc=0.3
        posterior_inc=0.15
        
        #Plot 95% CI for posterior results
        ax.plot((i+posterior_inc)*np.ones(3),intervals_posterior[column],marker='o', color='gray', label=label_ci_posterior)
        
        #Plot 95% CI for prior results
        ax.plot((i+prior_inc)*np.ones(3),intervals[column],marker='o', color='black', label=label_ci)
        
        #Plot posterior means
        ax.scatter([i+posterior_inc], df.loc['posterior_mean', column],marker='x', color='gray',s=100, label=label_posterior_mean)
        
        #Plot unposterior means
        ax.scatter([i+prior_inc], df.loc['prior_mean', column],marker='x', color='black',s=100, label=label_mean)
    
    #Plot observed trend
    ax.axhline(y=obs_warming_trend, label='Observed trend', color='gray', linestyle='dashed')
    
    
    ax.legend()
    ax.set_ylabel('May-September warming rate in Finland (°C/decade)')
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels, minor=False, rotation=45)
    cbar=fig.colorbar(sc)
    cbar.set_label('Weight')
    ax.set_title(title)
    if cut_warming_rate:
        ax.set_ylim([-1.25,3.3])
    fig.savefig(figname, dpi=150)
    
visualize(df_all_ssps, intervals_all_ssps, intervals_posterior_all_ssps, title='SSPs 1-1.9, 1-2.6 and 5-8.5 with 8 models', figname=figdir / 'scatter_all_ssps.png')    
visualize(df_all_models, intervals_all_models, intervals_posterior_all_models, title='SSPs  1-2.6 and 5-8.5 with 27 models', figname=figdir / 'scatter_all_models.png') 
visualize(df_all_models, intervals_all_models, intervals_posterior_all_models, title='SSPs  1-2.6 and 5-8.5 with 27 models', figname=figdir / 'scatter_all_models_cut_warming_rate.png',cut_warming_rate=True )  
# visualize(df_all_models)   

def plot_pdfs(df,ssps,kde_prior, kde_posterior, figname):
    fig,ax=plt.subplots(len(ssps),2)
    i=0
    j=0
    for ssp in ssps:
        j=0
        for method in ['','-scaled']:
            scen=ssp+method
            grid=np.linspace(-1,3, 10000)
            
            # Plot posterior histogram and kde
            df[scen][0:-2].hist(weights=df['weight'][0:-2], bins=8, density=True, ax=ax[i,j], alpha=0.5, color='orange',label='posterior histogram')
            ax[i,j].plot(grid,kde_posterior[scen].pdf(grid), color='orange',label='Posterior  KDE')
            
            # Plot unposterior histogram and kde
            df[scen][0:-2].hist(bins=8, density=True, ax=ax[i,j], alpha=0.5, color='blue',label='Prior histogram')
            ax[i,j].set_title(scen)
            j+=1
            
            
        i+=1
    ax[i-1,j-1].legend(bbox_to_anchor=(1.05, 1))
    fig.savefig(figname, dpi=150, bbox_inches='tight')

plot_pdfs(df_all_ssps, ssps3, kde_prior_all_ssps, kde_posterior_all_ssps, figdir / 'warming_rate_pdfs_all_ssps.png')
plot_pdfs(df_all_models, ssps2, kde_prior_all_models, kde_posterior_all_models, figdir / 'warming_rate_pdfs_all_models.png')

# Plot TCR distributions
fig, ax=plt.subplots(1,1)
tcr_grid=np.linspace(0,4,10000)
ax.plot(tcr_grid,tcr_norm.pdf(tcr_grid), label='IPCC AR6')
ax.plot(tcr_grid,kde_prior_all_ssps['TCR'].pdf(tcr_grid), label='Prior with all SSPs')
ax.plot(tcr_grid,kde_posterior_all_ssps['TCR'].pdf(tcr_grid), label='Posterior with all SSPs')
ax.plot(tcr_grid,kde_prior_all_models['TCR'].pdf(tcr_grid), label='Prior with all mdodels')
ax.plot(tcr_grid,kde_posterior_all_models['TCR'].pdf(tcr_grid), label='Posterior with all models')
ax.set_xlabel('Transient climate response °C')
ax.legend()
fig.savefig(figdir / 'TCR_pdfs.png', dpi=150)

#Plot scatter plot with TCR against warming rates
def plot_scatter_TCR_warming_rate(df, title, figname, cut_warming_rate=False):
    models=df.index[0:-8]
    fig, ax=plt.subplots(1,1)
    for scen in df.columns[0:-2]:
        ax.scatter(df.loc[models, 'TCR'],df.loc[models, scen],label=scen, s=5)
        
    ax.legend()
    ax.set_xlabel('Transient climate response °C')
    ax.set_ylabel('May-September warming rate°C/decade')
    ax.set_title(title)
    if cut_warming_rate:
        ax.set_ylim([-0.8, 4.1])
    fig.savefig(figname, dpi=150)
    
plot_scatter_TCR_warming_rate(df_all_models, 'SSPs  1-2.6 and 5-8.5 with 27 models', figdir / 'scatter_TCR_warmig_rate_all_models.png')
plot_scatter_TCR_warming_rate(df_all_models, 'SSPs  1-2.6 and 5-8.5 with 27 models', figdir / 'scatter_TCR_warmig_rate_all_models_cut_warming_rate.png', cut_warming_rate=True)
plot_scatter_TCR_warming_rate(df_all_ssps, 'SSPs 1-1.9, 1-2.6 and 5-8.5 with 8 models', figdir / 'scatter_TCR_warmig_rate_all_ssps.png')



#Save data into Excel 
with pd.ExcelWriter(outputdir/'processsed_cmip6_data.xlsx') as writer:
   
    # use to_excel function and specify the sheet_name and index
    # to store the dataframe in specified sheet
    df_all_ssps.to_excel(writer, sheet_name="All SSPs")
    df_all_models.to_excel(writer, sheet_name="All models")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%         Generate Data          %%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from utilities import *
from relabeler import *
from __future__ import division
import numpy as np


d = 1  # dimension of each time series
r = 1  # autoregressive order for each time series
T = 1000 # length of each time-series.  Note, each could be a different length.
m = d*r
K = inv(diag([10*np.ones((1,m))]))  # matrix normal hyperparameter (affects covariance of matrix)
M = np.zeros((d,m)) # matrix normal hyperparameter (mean matrix)

nu = d+2  # inverse Wishart degrees of freedom
meanSigma = 0.5*eye(d) # inverse Wishart mean covariance matrix
nu_delta = (nu-d-1)*meanSigma

numObj = 5 # number of time series
numStates = 9  # number of behaviors
for k in range(0,numStates):
    Sigma[k] = iwishrnd(nu_delta,nu);   # sample a covariance matrix
    
    if r==1: # if autoregressive order is 1, use some predefined dynamic matrices that cover the range of stable dynamics
        A[k] = (-1 +0.2*k)*eye(d)
    else:
        A[k] = sampleFromMatrixNormal(M,Sigma[k],K)  # otherwise, sample a random set of lag matrices (each behavior might not be very distinguishable!)


# Define feature matrix by sampling from truncated IBP:
F = sample_truncated_features_init(numObj,numStates,10);

# Define transition distributions:
p_self = 0.95
pi_z = ((1-p_self)/(numStates-1))*np.ones((numStates,numStates))
for ii in range(0,numStates):
    pi_z(ii,ii) = p_self

pi_init = np.ones((1,numStates))
pi_init = pi_init/np.sum(pi_init)
pi_s = np.ones((numStates,1))
dist_struct_tmp['pi_z'] = pi_z
dist_struct_tmp['pi_init'] = pi_init
dist_struct_tmp['pi_s'] = pi_s

for nn in range(0,numObj):
    
    Kz_inds = (F[nn,:]>0).nonzero
    
    pi_z_nn,pi_init_nn = transformDistStruct(dist_struct_tmp,Kz_inds)
    
    del Y 
    del X
    labels = np.zeros((1,T))
    P = np.cumsum(pi_init_nn)
    labels_temp = 1+np.sum(P[-1]*np.random.rand(1) > P)
    labels[0] = Kz_inds[labels_temp]
    tmp = mvnrnd(np.zeros((d,1)).T,Sigma[labels[1],r].T)
    x0 = tmp[:]
    x = x0
    
    for k in range(0,T):
        if k>1:
            P = np.cumsum(pi_z_nn[labels[k-1],:])
            labels[k] = 1+np.sum(P[-1]*np.random.rand(1) > P)
        Y[:,k] = A[labels(k)]*x + np.random.multivariate_normal(np.zeros((d,1)).T,Sigma[labels(k),1].T)
        X[:,k] = x;
        x = [[Y[:,k],x[1:-1-d]],:]
    

    
    data_struct[nn]['obs'] = Y
    data_struct[nn]['true_labels'] = labels
    



#clear model settings
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%      Set Model Params     %%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Set mean covariance matrix from data (instead of assuming knowledge of
# ground truth):

Ybig2 = [];
for ii in range(0,len(data_struct)):
    Ybig2 = np.concatenate(Ybig2,data_struct[ii]['obs'])

nu = d+2;
meanSigma = 0.75*np.cov(np.diff(Ybig2.T))

obsModelType = 'AR'
priorType = 'MNIW'

# Set hyperprior settings for Dirichlet and IBP
a_alpha = 1;
b_alpha = 1;
var_alpha = 1;
a_kappa = 100;
b_kappa = 1;
var_kappa = 100;
a_gamma = 0.1;
b_gamma = 1;

# The 'getModel' function takes the settings above and creates the
# necessary 'model' structure.
getModel

# Setting for inference:
settings['Ks'] = 1  # legacy parameter setting from previous code.  Do not change.
settings['Niter'] = 1000  # Number of iterations of the MCMC sampler  
settings['storeEvery'] = 1  # How often to store MCMC statistics
settings['saveEvery'] = 100  # How often to save (to disk) structure containing MCMC sample statistics
settings['ploton'] = 1  # Whether or not to plot the mode sequences and feature matrix while running sampler
settings['plotEvery'] = 10  # How frequently plots are displayed
settings['plotpause'] = 0



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%% Run IBP Inference %%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#close all
# Directory to which you want statistics stored.  This directory will be
# created if it does not already exist:
settings['saveDir'] = '../savedStats/BPARHMM/'; 

settings['formZInit'] = 1 # whether or not the sampler should be initialized with specified mode sequence.  (Experimentally, this seems to work well.)
settings['ploton'] = 1

# Number of initializations/chains of the MCMC sampler:
trial_vec = range(0,10)

for t in trial_vec:
    
    z_max = 0
    for seq in range(0,len(data_struct.items())):
        
        # Form initial mode sequences to simply block partition each
        # time series into 'Ninit' features.  Time series are given
        # non-overlapping feature labels:
        T = np.shape(data_struct[seq]['obs'],2)
        Ninit = 5
        init_blocksize = floor(T/Ninit)
        z_init = []
        for i in range(0,Ninit):
            z_init = np.concatenate(z_init,(i+1)*np.ones((1,init_blocksize)))
        
        z_init[Ninit*init_blocksize+1:T] = Ninit
        data_struct[seq]['z_init'] = z_init + z_max;
        
        z_max = np.max(data_struct[seq]['z_init'])


    settings['trial'] = t
    
    # Call to main function:
    IBPHMMinference(data_struct,model,settings)

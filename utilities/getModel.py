# Set Hyperparameters

del model

# Type of dynamical system:
model['obsModel']['type'] = obsModelType

if obsModelType=='AR':
    # Order of AR process:
    model['obsModel']['r'] = r
    m = d*r
else:
    m = d


# Type of prior on dynamic parameters. Choices include matrix normal
# inverse Wishart on (A,Sigma) and normal on mu ('MNIW-N'), matrix normal
# inverse Wishart on (A,Sigma) with mean forced to 0 ('MNIW'), normal on A,
# inverse Wishart on Sigma, and normal on mu ('N-IW-N'), and fixed A,
# inverse Wishart on Sigma, and normal on mu ('Afixed-IW-N').  NOTE: right
# now, the 'N-IW-N' option is only coded for shared A!!!
model['obsModel']['priorType'] = priorType

if priorType=='NIW':

        model['obsModel']['params']['M']  = np.zeros((d,1))
        model['obsModel']['params']['K'] =  kappa
        
elif priorType=='IW-N':
        # Mean and covariance for Gaussian prior on mean:
        model['obsModel']['params']['mu0'] = np.zeros((d,1))
        model['obsModel']['params']['cholSigma0'] = np.cholelsky(sig0*np.identity(d))
    
elif priorType=='MNIW':
        # Mean and covariance for A matrix:
        model['obsModel']['params']['M']  = np.zeros((d,m))

        # Inverse covariance along rows of A (sampled Sigma acts as
        # covariance along columns):
        model['obsModel']['params']['K'] =  K[0:m,0:m]
        
elif priorType=='MNIW-N':
        # Mean and covariance for A matrix:
        model['obsModel']['params']['M']  = np.zeros((d,m))

        # Inverse covariance along rows of A (sampled Sigma acts as
        # covariance along columns):
        model['obsModel']['params']['K'] =  K[0:m,0:m]

        # Mean and covariance for mean of process noise:
        model['obsModel']['params']['mu0'] = np.zeros((d,1))
        model['obsModel']['params']['cholSigma0'] = np.cholelsky(sig0*np.identity(d))

elif priorType=='N-IW-N':
        # Mean and covariance for A matrix:
        model['obsModel']['params']['M']  = np.zeros((d,m))
        model['obsModel']['params']['Lambda0_A'] = np.linalg.inv(np.kron(np.linalg.inv(K),meanSigma))

        # Mean and covariance for mean of process noise:
        model['obsModel']['params']['mu0'] = np.zeros((d,1))
        model['obsModel']['params']['cholSigma0'] = np.cholesky(sig0*np.identity(d))
        
elif priorType=='Afixed-IW-N':
        # Set fixed A matrix:
        model['obsModel']['params']['A'] = A_shared
        
        # Mean and covariance for mean of process noise:
        model['obsModel']['params']['mu0'] = np.zeros((d,1))
        model['obsModel']['params']['cholSigma0'] = np.cholelsky(sig0*np.identity(d))
        
elif priorType=='ARD':        # Gamma hyperprior parameters for prior on precision parameter:
        model['obsModel']['params']['a_ARD'] = a_ARD
        model['obsModel']['params']['b_ARD'] = b_ARD
        
        # Placeholder for initializeStructs. Can I get rid of this?
        model['obsModel']['params']['M']  = np.zeros((d,m))

        # Mean and covariance for mean of process noise:
        model['obsModel']['params']['zeroMean'] = 1

        
# Degrees of freedom and scale matrix for covariance of process noise:
model['obsModel']['params']['nu'] = nu #d + 2
model['obsModel']['params']['nu_delta'] = (model['obsModel']['params']['nu']-d-1)*meanSigma

if obsModelType=='SLDS':
    # Degrees of freedom and scale matrix for covariance of measurement noise:
    model['obsModel']['y_params']['nu'] = nu_y #dy + 2
    model['obsModel']['y_params']['nu_delta'] = (model['obsModel']['y_params']['nu']-dy-1)*meanR
    
    model['obsModel']['y_priorType'] = y_priorType
    
    if model['obsModel']['y_priorType']=='NIW':
            model['obsModel']['y_params']['M']  = np.zeros((dy,1))
            model['obsModel']['y_params']['K'] =  kappa_y
            
    elif y_priorType=='IW-N':
            # Mean and covariance for Gaussian prior on mean:
            model['obsModel']['y_params']['mu0'] = mu0_y #zeros(dy,1)
            model['obsModel']['y_params']['cholSigma0'] = np.cholesky(sig0_y*inp.dentity(dy))

    
    # Fixed measurement matrix:
    model['obsModel']['params']['C'] = np.concatenate(np.identity(dy),np.zeros((dy,d-dy)))
    
    # Initial state covariance:
    model['obsModel']['params']['P0'] = P0*np.identity(d)

# Always using DP mixtures emissions, with single Gaussian forced by
# Ks=1...Need to fix.
model['obsModel']['mixtureType'] = 'infinite'

# Sticky HDP-HMM parameter settings:
model['HMMmodel']['params']['a_alpha']=a_alpha  # affects \pi_z
model['HMMmodel']['params']['b_alpha']=b_alpha
model['HMMmodel']['params']['var_alpha']=var_alpha
model['HMMmodel']['params']['a_kappa']=a_kappa  # affects \pi_z
model['HMMmodel']['params']['b_kappa']=b_kappa
model['HMMmodel']['params']['var_kappa']=var_kappa
model['HMMmodel']['params']['a_gamma']=a_gamma  # global expected # of HMM states (affects \beta)
model['HMMmodel']['params']['b_gamma']=b_gamma

numObj = len(data_struct.items())
harmonic = 0
for n in range(0,len(data_struct.items())):
    harmonic = harmonic + 1/n

model['HMMmodel']['params']['harmonic'] = harmonic
if exist('Ks'):
    if Ks>1:
        model['HMMmodel']['params']['a_sigma'] = 1
        model['HMMmodel']['params']['b_sigma'] = 0.01

else:
    Ks = 1

if exist('Kr'):
    if Kr > 1:
        model['HMMmodel']['params']['a_eta'] = 1
        model['HMMmodel']['params']['b_eta'] = 0.01


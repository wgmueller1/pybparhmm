import numpy as np

def sample_theta_extra(theta,obsModel,Kextra):

    prior_params = obsModel['params']

    if obsModel.priorType!='MNIW':
        Exception('Only coded for MNIW prior')


    nu = prior_params['nu']
    nu_delta = prior_params['nu_delta']

    invSigma = theta['invSigma']
    A = theta['A']

    tmp1,tmp2,Kz,Ks = np.shape(invSigma)

    K = prior_params['K']
    M = prior_params['M']

    for kz in range(Kz,Kz+Kextra):
        for ks in range(0,Ks):
            
            Sxx = K
            SyxSxxInv = M
            Sygx = 0
            
            # Sample Sigma given s.stats
            sqrtSigma,sqrtinvSigma = randiwishart(Sygx + nu_delta,nu);
            invSigma[:,:,kz,ks] = sqrtinvSigma.T*sqrtinvSigma
            
            # Sample A given Sigma and s.stats
            cholinvSxx = np.linalg.cholesky(np.linalg.inverse(Sxx))
            A[:,:,kz,ks] = sampleFromMatrixNormal(SyxSxxInv,sqrtSigma,cholinvSxx)
            

    theta['invSigma'] = invSigma
    theta['A'] =  A

    return theta
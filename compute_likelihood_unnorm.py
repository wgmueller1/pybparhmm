def compute_likelihood_unnorm(data_struct,theta,obsModelType,Kz_inds,Kz,Ks):
    options = {'Gaussian' : gaussian_loglike,'AR' : arslds_loglike,'SLDS' : arslds_loglike,'Multinomial' : multinomial_loglike}
    return log_likelihood

def gaussian_loglike():
        invSigma = theta['invSigma']
        mu = theta['mu']
        
        T = data_struct['obs'].shape[1]
        dimu = data_struct['obs'].shape[0]
        
        log_likelihood = float("-inf")*np.ones([Kz,Ks,T])
        for kz in Kz_inds:
            for ks in range(0,Ks):
                
                cholinvSigma = np.linalg.cholesky(invSigma[:,:,kz,ks])
                dcholinvSigma = np.diag(cholinvSigma)
                
                u = cholinvSigma*(data_struct['obs'] - mu[:,kz*np.ones([1,T]),ks])
                
                log_likelihood[kz,ks,:] = -0.5*np.sum(u**2,axis=0) + np.sum(log(dcholinvSigma))
    
        return log_likelihood
 
def arslds_loglike():
        invSigma = theta['invSigma']
        A = theta['A']
        X = data_struct['X']
        
        T = data_struct['obs'].shape[1]
        dimu = data_struct['obs'].shape[0]
        
        log_likelihood = float("-inf")*np.ones([Kz,Ks,T])
        if 'mu' in theta.keys():
            
            mu = theta['mu']
            
            for kz in Kz_inds:
                for ks in range(0,Ks):
                    cholinvSigma = np.linalg.cholesky(invSigma[:,:,kz,ks])
                    dcholinvSigma = np.diag(cholinvSigma)
                    u = cholinvSigma*(data_struct['obs'] - A[:,:,kz,ks]*X-mu[:,kz*np.ones([1,T]),ks]);
                    log_likelihood[kz,ks,:] = -0.5*np.sum(u**2,axis=0) + np.sum(log(dcholinvSigma));

        else:
            
            for kz in Kz_inds:
                for ks in range(0,Ks):
                    cholinvSigma = np.linalg.cholesky(invSigma[:,:,kz,ks])
                    dcholinvSigma = np.diag(cholinvSigma)
                    u = cholinvSigma*(data_struct['obs'] - A[:,:,kz,ks]*X)
                    log_likelihood[kz,ks,:] = -0.5*np.sum(u**2,axis=0) + np.sum(log(dcholinvSigma))
                    
        return log_likelihood
  
def multinomial_loglike():
    log_likelihood = np.log(theta['p'][:,:,data_struct['obs']])
    return log_likelihood
       

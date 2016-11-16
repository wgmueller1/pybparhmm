def compute_likelihood(data_struct,theta,obsModelType,Kz_inds,Kz,Ks):
    options = {'Gaussian' : gaussian_like(),'AR' : arslds_like(),'SLDS' : arslds_like(),'Multinomial' : multinomial_like()}
    #[theta,Ustats,stateCounts,data_struct,model,S] =  options[obsModelType]
    return [theta,Ustats,stateCounts,data_struct,model,S]
 
def gaussian_like():
    invSigma = theta['invSigma'];
    mu = theta['mu'];
        
    T = data_struct['obs'].shape[1];
    dimu = data_struct['obs'].shape[0];
        
    log_likelihood = float("-inf")*np.ones([Kz,Ks,T])
    for kz in Kz_inds:
        for ks in range(0,Ks):
                
                cholinvSigma = np.linalg.cholesky(invSigma[:,:,kz,ks])
                dcholinvSigma = np.diag(cholinvSigma)
                
                u = cholinvSigma*(data_struct['obs'] - mu[:,kz*np.ones([1,T]),ks])
                
                log_likelihood[kz,ks,:] = -0.5*np.sum(u**2,axis=0) + np.sum(log(dcholinvSigma))
        normalizer = np.max(np.max(log_likelihood,axis=0),axis=1)
        log_likelihood = log_likelihood - normalizer[np.ones([Kz,1]),np.ones([Ks,1]),:]
        likelihood = exp(log_likelihood)
        
    normalizer = normalizer - (dimu/2.0)*log(2*pi)
    return likelihood,normalizer
 
def arslds_like():
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
                    u = cholinvSigma*(data_struct['obs'] - A[:,:,kz,ks]*X);
                    log_likelihood[kz,ks,:] = -0.5*np.sum(u**2,axis=0) + np.sum(log(dcholinvSigma));
                    
        
       
        normalizer = np.max(np.max(log_likelihood,axis=0),axis=1);
        log_likelihood = log_likelihood - normalizer[np.ones([Kz,1]),np.ones([Ks,1]),:];
        likelihood = exp(log_likelihood);
       
        normalizer = normalizer - (dimu/2.0)*log(2*pi);
        return likelihood,normalizer
  
def multinomial_like():
    likelihood = theta['p'][:,:,data_struct['obs']];
    normalizer = np.zeros([1,data_struct['obs'].shape[1]]);
    return likelihood,normalizer
       
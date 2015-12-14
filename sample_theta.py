def sample_theta(theta,Ustats,obsModel,Kextra):

prior_params = obsModel'params']
if obsModel['type']=='Multinomial':
        
    p = theta['p']
    store_counts = Ustats['card']
    alpha_vec = prior_params['alpha']
        
    Kz,Ks = store_counts.shape
        
    for kz in range(0,Kz):
        for ks in range(0,Ks):
            #doublecheck
            p[kz,ks,:] = randdirichlet([alpha_vec.T+store_counts[:,kz,ks]]).T

        
    theta['p'] = p
        
 elif obsModel['type'] in ['Gaussian','AR','SLDS']:
        
        theta = sample_theta_submodule(theta,Ustats,obsModel['priorType'],prior_params,Kextra)
        
        if obsModel['type']=='SLDS':
            y_prior_params = obsModel['y_params']
            #doublecheck
            theta['theta_r'] = sample_theta_submodule(theta['theta_r'],Ustats['Ustats_r'],\
                obsModel['y_priorType'],y_prior_params,[])
        
return theta


def sample_theta_submodule(theta,Ustats,priorType,prior_params,Kextra):

    nu = prior_params['nu']
    nu_delta = prior_params['nu_delta']
    store_card = Ustats['card']

    if store_card.shape[0]==1:
        store_card = store_card.T
    #double check this is hstack or vstack
    store_card = [store_card; np.zeros([Kextra,store_card.shape[1]])]
    Kz,Ks = store_card.shape

    if priorType=='MNIW':
        
        invSigma = theta['invSigma']
        A = theta['A']
        
        store_XX = Ustats['XX']
        store_YX = Ustats['YX']
        store_YY = Ustats['YY']
        store_sumY = Ustats['sumY']
        store_sumX = Ustats['sumX']
        
        K = prior_params['K']
        M = prior_params['M']
        MK = prior_params['M']*prior_params['K']
        MKM = MK*prior_params['M'].T
        
        for kz in range(0,Kz):
            for ks in range(0,Ks):
                
                if store_card[kz,ks]>0:
                    
                    # Given X, Y get sufficient statistics
                    Sxx       = store_XX[:,:,kz,ks] + K
                    Syx       = store_YX[:,:,kz,ks] + MK
                    Syy       = store_YY[:,:,kz,ks] + MKM
                    SyxSxxInv = Syx/Sxx
                    Sygx      = Syy - SyxSxxInv*Syx.T
                    Sygx = (Sygx + Sygx.T)/2
                    
                else:
                    Sxx = K
                    SyxSxxInv = M
                    Sygx = 0
                
                # Sample Sigma given s.stats
                sqrtSigma,sqrtinvSigma = randiwishart(Sygx + nu_delta,nu+store_card[kz,ks])
                invSigma[:,:,kz,ks] = sqrtinvSigma.T*sqrtinvSigma
                
                # Sample A given Sigma and s.stats
                cholinvSxx = np.linalg.cholesky(np.linalg.inv(Sxx))
                A[:,:,kz,ks] = sampleFromMatrixNormal(SyxSxxInv,sqrtSigma,cholinvSxx)
                
        
        theta['invSigma'] = invSigma
        theta['A'] =  A

   elif priorType=='NIW':
        
        invSigma = theta['invSigma']
        mu = theta['mu']
        
        store_YY = Ustats['YY']
        store_sumY = Ustats['sumY']
        
        K = prior_params['K']
        M = prior_params['M']
        MK = prior_params['M']*prior_params['K']
        MKM = MK*prior_params['M'].T
        
        for kz in range(0,Kz):
            for ks in range(0,Ks):
                
                if store_card[kz,ks]>0:
                    
                    ## Given X, Y get sufficient statistics
                    Sxx       = store_card[kz,ks] + K
                    Syx       = store_sumY[:,kz,ks] + MK
                    Syy       = store_YY[:,:,kz,ks] + MKM
                    SyxSxxInv = Syx/Sxx
                    Sygx      = Syy - SyxSxxInv*Syx.T
                    Sygx = (Sygx + Sygx.T)/2
                    
                else:
                    Sxx = K
                    SyxSxxInv = M
                    Sygx = 0
                    
                
                # Sample Sigma given s.stats
                sqrtSigma,sqrtinvSigma = randiwishart(Sygx + nu_delta,nu+store_card[kz,ks])
                invSigma[:,:,kz,ks] = sqrtinvSigma.T*sqrtinvSigma
                
                # Sample A given Sigma and s.stats
                cholinvSxx = np.linalg.cholesky(np.linalg.inv(Sxx))
                mu[:,kz,ks] = sampleFromMatrixNormal(SyxSxxInv,sqrtSigma,cholinvSxx)

        theta['invSigma'] = invSigma
        theta['mu'] =  mu

    elif priorType=='MNIW-N':
            
            invSigma = theta['invSigma']
            A = theta['A']
            mu = theta['mu']
            
            store_XX = Ustats['XX']
            store_YX = Ustats['YX']
            store_YY = Ustats['YY']
            store_sumY = Ustats['sumY']
            store_sumX = Ustats['sumX']
            
            # If MNIW-N, K and M are as in MNIW. If IW-N, K=1 and M=0.
            K = prior_params['K']
            M = prior_params['M']
            MK = prior_params['M']*prior_params['K']
            MKM = MK*prior_params['M'].T
            
            if 'numIter' not in prior_params.keys():
                prior_params.numIter = 50
            
            numIter = prior_params['numIter']
            
            mu0 = prior_params['mu0']
            cholSigma0 = prior_params['cholSigma0']
            Lambda0 = np.linalg.inv(prior_params['cholSigma0'].T*prior_params['cholSigma0'])
            theta0 = Lambda0*prior_params['mu0']
            
            dimu = nu_delta.shape[0]
            
            for kz in range(0,Kz):
                for ks in range(0,Ks):
                    
                    if store_card[kz,ks]>0:  #**
                        for n in range(0,numIter):
                            
                            ## Given X, Y get sufficient statistics
                            Sxx       = store_XX[:,:,kz,ks] + K
                            Syx       = store_YX[:,:,kz,ks] + MK - mu[:,kz,ks]*store_sumX[:,kz,ks].T
                            Syy       = store_YY[:,:,kz,ks] + MKM \
                                - mu[:,kz,ks]*store_sumY[:,kz,ks].T - store_sumY[:,kz,ks]*mu[:,kz,ks].T + \
                                 store_card[kz,ks]*mu[:,kz,ks]*mu[:,kz,ks].T
                            SyxSxxInv = Syx/Sxx
                            Sygx      = Syy - SyxSxxInv*Syx.T
                            Sygx = (Sygx + Sygx.T)/2
                            
                            # Sample Sigma given s.stats
                            sqrtSigma,sqrtinvSigma = randiwishart(Sygx + nu_delta,nu+store_card[kz,ks])
                            invSigma(:,:,kz,ks) = sqrtinvSigma.T*sqrtinvSigma
                            
                            # Sample A given Sigma and s.stats
                            cholinvSxx = np.linalg.chol(np.linalg.inverse(Sxx))
                            A[:,:,kz,ks] = sampleFromMatrixNormal(SyxSxxInv,sqrtSigma,cholinvSxx)
                            
                            # Sample mu given A and Sigma
                            Sigma_n = inv(Lambda0 + store_card(kz,ks)*invSigma[:,:,kz,ks])
                            mu_n = Sigma_n*(theta0 + invSigma(:,:,kz,ks)*(store_sumY[:,kz,ks]-A[:,:,kz,ks]\
                                *store_sumX[:,kz,ks]))
                            #doublecheck
                            mu[:,kz,ks] = mu_n + np.linalg.cholesky(Sigma_n).T*np.random.standard_normal(dimu,1)
                    
                    else:
                        Sxx = K
                        SyxSxxInv = M
                        Sygx = 0
                        
                        sqrtSigma,sqrtinvSigma = randiwishart(nu_delta,nu)
                        invSigma[:,:,kz,ks] = sqrtinvSigma.T*sqrtinvSigma
                        
                        cholinvK = np.linalg.cholesky(np.linalg.inv(K))
                        A[:,:,kz,ks] = sampleFromMatrixNormal(M,sqrtSigma,cholinvK)
                        mu[:,kz,ks] = mu0 + cholSigma0.T*np.random.standard_normal((dimu,1))

            theta['invSigma'] = invSigma
            theta['A'] = A
            theta['mu'] =  mu
            
    elif priorType=='IW-N':

        invSigma = theta['invSigma']
        mu = theta['mu']
        

        store_YY = Ustats['YY']
        store_sumY = Ustats['sumY']
    
        
        if 'numIter' not in prior_params.keys():
            prior_params['numIter'] = 50
        
        numIter = prior_params['numIter']
        
        mu0 = prior_params['mu0']
        cholSigma0 = prior_params['cholSigma0']
        Lambda0 = np.linalg.inv(prior_params['cholSigma0'].T*prior_params['cholSigma0'])
        theta0 = Lambda0*prior_params['mu0']
        
        dimu = nu_delta.shape[0]
        
        for kz in range(0,Kz):
            for ks in range(0,Ks):
                
                if store_card[kz,ks]>0:  #**
                    for n in range(0,numIter):
                        
                        ## Given X, Y get sufficient statistics
                        Syy       = store_YY[:,:,kz,ks] + \
                            - mu[:,kz,ks]*store_sumY[:,kz,ks].T - store_sumY[:,kz,ks]*mu[:,kz,ks].T +\
                             store_card[kz,ks]*mu[:,kz,ks]*mu[:,kz,ks].T
                        Sygx = (Syy + Syy.T)/2
                        
                        # Sample Sigma given s.stats
                        sqrtSigma,sqrtinvSigma = randiwishart(Sygx + nu_delta,nu+store_card[kz,ks])
                        invSigma[:,:,kz,ks] = sqrtinvSigma.T*sqrtinvSigma;
                        
                        # Sample A given Sigma and s.stats
                        cholinvSxx = np.linalg.cholesky(np.linalg.inv(Sxx))
                        A[:,:,kz,ks] = sampleFromMatrixNormal(SyxSxxInv,sqrtSigma,cholinvSxx)
                        
                        # Sample mu given A and Sigma
                        Sigma_n = np.linalg.inv(Lambda0 + store_card[kz,ks]*invSigma[:,:,kz,ks])
                        mu_n = Sigma_n*(theta0 + invSigma[:,:,kz,ks]*(store_sumY[:,kz,ks]-A[:,:,kz,ks]*store_sumX[:,kz,ks]))
                        
                        mu[:,kz,ks] = mu_n + np.linalg.cholesky(Sigma_n).T*np.random.standard_normal((dimu,1))
                        
                else:
                    Sxx = K
                    SyxSxxInv = M
                    Sygx = 0
                    
                    sqrtSigma,sqrtinvSigma = randiwishart(nu_delta,nu)
                    invSigma[:,:,kz,ks] = sqrtinvSigma.T*sqrtinvSigma
                    
                    cholinvK = np.cholesky(np.linalg.inv(K))
                    A[:,:,kz,ks] = sampleFromMatrixNormal(M,sqrtSigma,cholinvK)
                    
                    mu[:,kz,ks] = mu0 + cholSigma0.T*np.random.standard_normal(dimu,1)
        
        theta['invSigma'] = invSigma
        theta['A'] = A
        theta['mu'] =  mu







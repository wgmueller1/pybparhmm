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
                        mu_n = Sigma_n*(theta0 + invSigma[:,:,kz,ks]*store_sumY[:,kz,ks]))
                        
                        mu[:,kz,ks] = mu_n + np.linalg.cholesky(Sigma_n).T*np.random.standard_normal((dimu,1))
                        
                else:
    
                    
                    sqrtSigma,sqrtinvSigma = randiwishart(nu_delta,nu)
                    invSigma[:,:,kz,ks] = sqrtinvSigma.T*sqrtinvSigma
                    

                    mu[:,kz,ks] = mu0 + cholSigma0.T*np.random.standard_normal(dimu,1)
        
        theta['invSigma'] = invSigma
        theta['A'] = A
        theta['mu'] =  mu

    elif priorType=='IW-N-tiedwithin':
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

        dimu=nu_delta.shape[0]
        for kz in range(0,numIter):
            store_invSigma = invSigma[:,:,kz,1]
            for n in range(0,numIter):

                for ks in range(0,Ks):
                    if store_card(kz,ks)>0:
                        Sigma_n = np.linalg.inv(Lambda0 + store_card[kz,ks]*store_invSigma)
                        mu_n = Sigma_n*(theta0 + store_invSigma*store_sumY[:,kz,ks])
                        mu[:,kz,ks] = mu_n + np.linalg.cholesky(Sigma_n).T*np.random.standard_normal((dimu,1))
                    else:
                        mu[:,kz,ks] = mu0 + cholSimga0.T*np.random.standard_normal((dimu,1))

                #Given Y get sufficient statistics
                store_card_kz = store_card(kz,:);
                #need to double check squeeze function
                squeeze_mu_kz = squeeze(mu[:,kz,:])
                muY = squeeze_mu_kz*squeeze(store_sumY[:,kz,:].T)
                muN = (np.matlib.repmat(store_card_kz,(dimu,1))*squeeze_mu_kz)*squeeze_mu_kz.T
                Syy = np.sum(store_YY[:,:,kz,:],axis=3) - muY - muY.T+muN
                Syy = (Syy+Syy.T)/2

                # Sample Sigma given s.stats
                sqrtSigma,sqrtinvSigma = randiwishart(Syy+nu_delta,nu+np.sum(store_card_kz))
                store_invSigma = sqrtSigma.T*sqrtSigma
        invSigma[:,:,kz,0:Ks]=np.matlib.repmat(store_invSigma,np.array([1,1,1,Ks]))

    theta['invSigma'] = invSigma
    theta['mu'] = mu

elif priorType in ['N-IW-N','Afixed-IW-N','ARD']:

    invSigma = theta['invSigma']
    A = theta['A']
    mu = theta['mu']

    store_XX = Ustats['XX']
    store_YX = Ustats['YX']
    store_YY = Ustats['YY']
    store_sumY = Ustats['sumY']
    store_sumX = Ustats['sumX']

    if 'numIter' not in prio_params.keys():
        prior_params['numIter']=50

    numIter = prior_params['numIter']

    if 'zeroMean' in prior_params.keys():
        mu0 = prior_params['mu0']
        cholSigma0 = prior_params['cholSigma0']
        Lambda0 = np.linalg.inv(prior_params['cholSigma0'].T*prior_params['cholSigma0'])
        theta0 = Lambda0*prior_params['mu0']

    r = A.shape[1]/A.shape[0]

    if priorType=='N-IW-N':
        M = prior_params['M']
        Lambda0_A = prior_params['Lambda0_A']
        theta0_A = Lambda0_A*M[:]
        dim_vecA = M.size

        XinvSigmaX = np.zeros((dim_vecA,dim_vecA))
        XinvSigmay = np.zeros((dim_vecA,1))

    elif priorType=='ARD':
        M=np.zeros(A[:,:,1,1].shape)
        theta0_A = np.zeros((A.size,1))
        dim_vecA = M.size

        a_ARD = prior_params['a_ARD']
        b_ARD = prior_params['b_ARD']

        if r==1:
            numHypers = M.shape[0]
        else:
            numHypers =r

        ARDhypers = np.ones((numHypers,Kz,Ks))
        posInds = np.where(store_card>0)
        if x.size>0:
            ARDhypers[:,0:mu.shape[1]]=theta['ARDhypers']

    dim = nu_delta.shape[0]
    for n in range(0,numIter):
        for kz in range(0,Kz):
            for ks in range(0,Ks):
                if store_card[kz,ks]>0:
                    #Sample Sigma given A, mu, and s.stats
                    S= store_YY[:,:,kz,ks] + A[:,:,kz,ks]*store_XX[:,:,kz,ks]*A[:,:,kz,ks].T\
                    -A[:,:,kz,ks]*store_YX[:,:,kz,ks].T - store_YX[:,:,kz,ks]*A[:,:,kz,ks].T\
                    -mu[:,kz,ks]*(store_sumY[:,kz,ks]-A[:,:,kz,ks]*store_sumX[:,kz,ks]).T-\
                    (store_sumY[:,kz,ks]-A[:,:,kz,ks]*store_sumX[:,kz,ks])*mu[:,kz,ks].T\
                    +store_card[kz,ks]*mu[:,kz,ks]*mu[:,kz,ks].T
                    S=0.5*(S+S.T)

                    sqrtSigma,sqrtinvSigma=randiwishart(S+nu_delta,nu+store_card[kz,ks])
                    invSigma[:,:,kz,ks] = sqrtinvSigma.T*sqrtinvSigma

                    if prior_params=='zeroMean':
                        mu[:,kz,ks]=np.zeros((dimu,1))
                    else:
                        #Sample mu given A, Sigma, and s.stats
                        Sigma_n = np.inv(Lambda0 + store_card(kz,ks)*invSigma[:,:,:kz,ks])
                        mu_n = Sigma_n *(theta0 + invSigma[:,:,kz,ks]*(store_sumY[:,kz,ks]-A[:,:,kz,ks]*store_sumX[:,kz,ks]))
                        mu[:,kz,ks] = mu_n +np.cholesky(Sigma_n).T*randn(dimu,1)

                    if priorType='N-IW-N':
                        XinvSigmaX=XinvSigmaX+np.kron(store_XX[:,:,kz,ks],invSigma[:,:,kz,ks])
                        temp = invSigma[:,:,kz,ks]*(store_YX[:,:,kz,ks]-mu[:,kz,ks]*store_sumX[:,kz,ks].T)
                        XinvSigmay = XinvSigmay + temp[:] #since A is shared, grow for all data, not just kz,ks
                    elif priorType=='ARD': 
                        XinvSigmaX = np.kron(store_XX[:,:,kz,ks],invSigma[:,:,kz,ks])
                        XinvSigmay = invSigma[:,:,kz,ks]*(store_YX[:,:,kz,ks]-mu[:,kz,ks]*store_sumX[:,kz,ks].T)
                        XinvSigmay = XinvSigmay[:]
                        ARDhypers_kzks = ARDhypers[:,kz,ks]
                        if r == 1:
                            numObsPerHyper = numRow
                        else:
                            numObsPerHyper = numRow*(numCol/r)
                            
                        ARDhypers_kzks = ARDhypers_kzks[:,np.ones([1,numObsPerHyper])].T
                        ARDhypers_kzks = ARDhypers_kzks[:]
                        Lambda0_A = np.diag(ARDhypers_kzks)
                            
                        Sigma_A = np.linalgo.inv(Lambda0_A + XinvSigmaX)
                        mu_A = Sigma_A*(theta0_A + XinvSigmay)
                        vecA = mu_A + np.linalg.cholesky(Sigma_A).T*randn(dim_vecA,1)
                        #A(:,:,kz,ks) = reshape(vecA,size(M))
                           
                        #AA = sum(A(:,:,kz,ks).*A(:,:,kz,ks),1);
                        #if r>1:
                        #    AA = reshape(AA,[numCol/r r]);
                        #    AA = sum(AA,1);
                          
                        aa = np.zeros([1,numHypers])
                        for ii in range(0,numHypers):
                            aa(ii) = randgamma(a_ARD + numObsPerHyper/2);
                            #aa(ii) = randgamma(a_ARD + numRow/2);
                            
                        ARDhypers[:,kz,ks] =  aa / (b_ARD + AA/2);
                      
                        
      
        
    elif priorType=='IW':

        invSigma = theta['invSigma']
        
        store_YY = Ustats['YY']
        store_sumY = Ustats['sumY']
        
        dimu = np.shape(nu_delta)[0]
        
        for kz in range(0,Kz):
            for ks in range(0,Ks):
                
                if store_card[kz,ks]>0:  #**
                    
                    # Given X, Y get sufficient statistics
                    Syy  = store_YY[:,:,kz,ks]
                    Sygx = (Syy + Syy.T)/2
                    
                    # Sample Sigma given s.stats
                    sqrtSigma,sqrtinvSigma = randiwishart(Sygx + nu_delta,nu+store_card[kz,ks])
                    invSigma(:,:,kz,ks) = sqrtinvSigma.T*sqrtinvSigma
                    
                else:
                    sqrtSigma,sqrtinvSigma = randiwishart(nu_delta,nu);
                    invSigma[:,:,kz,ks] = sqrtinvSigma.T*sqrtinvSigma;
 
        theta['invSigma'] = invSigma
        
    return theta








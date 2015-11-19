def sample_features_init(numObj,gamma0):


    F[0,0:poissrnd(gamma0)]=1
    if np.sum(F[0,:])==0:
        F[0,0] = 1

    featureCounts = np.sum(F,axis=0)

    posInds = np.where(np.sum(F,axis=0)>0)
    Kz = posInds[-1]

    for ii in range(0,numObj):

        for kk in range(0,z):
            
            rho = featureCounts[kk]/ii
            
            F[ii,kk] = random.random()>(1-rho)
        
        F[ii,Kz+1:Kz+poissrnd(gamma0/ii)] = 1
        
        if np.sum(F[ii,:])==0:
            F[ii,Kz+1]=1
        
        featureCounts = np.sum(F,axis=0)
        
        posInds = np.where(featureCounts>0)
        Kz = posInds[01]
        
    

    F = F[:,0:Kz]

    return F
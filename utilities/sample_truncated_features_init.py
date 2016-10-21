def sample_truncated_features_init(numObj,Kz,gamma0):

    F = np.zeros((numObj,Kz))
    featureCounts = np.sum(F,axis=1)
    Fik_prev = 0

    for ii in range(0,numObj):

        for kk in range(0,Kz):
            
            rho = (featureCounts[kk] + gamma0/Kz)/(ii + gamma0/Kz)
            
            if rho>1:
                F[ii,kk] = np.logical_not(Fik_prev)
            else:
                sample_set = np.concatenate(Fik_prev,np.logical_not(Fik_prev))
                ind = np.random.random(1)>(1-rho)
                F(ii,kk) = sample_set[ind]

            
            F(ii,kk) = np.random.random(1)>(1-rho)
            
            featureCounts(kk) = featureCounts[kk]+F[ii,kk]       

def reallocate_states(F,dist_struct,theta,priorType):

    numObj,Kz = F.shape
    
    F,sort_ind = lof(F)
    
    posInds = np.where(np.sum(F)>0)
    sort_ind = sort_ind[0:posInds[-1]]
    F = F[:,0:posInds[-1]]
    for ii in range(0,numObj):
        dist_struct[ii]['pi_z'] = dist_struct[ii]['pi_z'][sort_ind,sort_ind]
        dist_struct[ii]['pi_s'] = dist_struct[ii]['pi_s'][sort_ind,:]
        dist_struct[ii]['pi_init'] = dist_struct[ii]['pi_init'][sort_ind]

    def prior1(theta):
        theta['invSigma'] = theta['invSigma'][:,:,sort_ind,:]
        theta['A'] =  theta['A'][:,:,sort_ind,:]
        return theta
        
    def prior2(theta):
        theta['invSigma'] = theta['invSigma'][:,:,sort_ind,:]
        theta['mu'] =  theta['mu'][:,1:Kz,:]
        return theta

    def prior3(theta):
        theta['invSigma'] = theta['invSigma'][:,:,sort_ind,:]
        theta['A'] = theta['A'][:,:,1:Kz,:]
        theta['mu'] =  theta['mu'][:,1:Kz,:]
        return theta     
   
    def prior4(theta):
        theta['invSigma'] = theta['invSigma'][:,:,sort_ind,:]
        theta['A'] = theta['A'][:,:,sort_ind,:]
        theta['mu'] =  theta['mu'][:,sort_ind,:]
        theta['ARDhypers'] = theta['ARDhypers'][:,sort_ind,:]
        return theta
        

    def prior5(theta):
        theta['invSigma'] = theta['invSigma'][:,:,sort_ind,:]
        return theta
	options = {'MNIW' : prior1,
                'NIW' : prior2,
                'IW-N' : prior2,
                'IW-N-tiedwithin' : prior2,
                'MNIW-N':prior3,
                'N-IW-N':prior3,
                'Afixed-IW-N':prior3,
                'ARD':prior4,
                'IW':prior5
                }
    theta=options[priorType]()
    return dist_struct,theta






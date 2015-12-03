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

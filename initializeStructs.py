def initializeStructs(F,model,data_struct,settings):

    Kz = F.shape[1]
    Ks = settings['Ks']

    prior_params = model['obsModel']['params']
    if blockSize not in data_struct[0].keys():
        data_struct[1].blockSize = []
    

    if model['obsModel']['type']=='Gaussian':

            dimu = data_struct[0]['obs'].shape[0]
            
            for ii in range(0,length(data_struct)):
                if np.size(data_struct[ii]['blockSize'])==0:
                    data_struct[ii]['blockSize'] = np.ones([1,data_struct[ii]['obs'].shape[1]])
                data_struct[ii]['blockEnd'] = np.cumsum(data_struct[ii]['blockSize'])
    

            theta = {'invSigma':np.zeros([dimu,dimu,Kz,Ks]),'mu':np.zeros([dimu,Kz,Ks])}  
            Ustats = {'card':np.zeros([Kz,Ks]),'YY':np.zeros([dimu,dimu,Kz,Ks]),'sumY':np.zeros([dimu,Kz,Ks])}
            
     elif model['obsModel']['type']=='Multinomial':

            for ii range(0,length(data_struct)):
                if data_struct[ii]['obs'].shape[0]>1
                    raise ValueError('not multinomial obs')
                if np.size(data_struct[ii]['blockSize'])==0:
                    data_struct[ii]['blockSize'] = np.ones([1,data_struct[ii]['obs'][1])])
                data_struct[ii]['blockEnd'] = np.cumsum(data_struct[ii]['blockSize']
            
            data_struct[0]['numVocab'] = len(prior_params['alpha']
            
            theta = {'p':np.zeros([Kz,Ks,data_struct[0]['numVocab'])}
            Ustats = {'card':np.zeros([data_struct[0]['numVocab'],Kz,Ks])}

     elif model['obsModel']['type']=='AR' or model['obsModel']['type']=='SLDS':

            if settings['Ks']!=1
                raise ValueError('Switching linear dynamical models only defined for Gaussian process noise, not MoG')
            
                if model['obsModel']['priorType']=='MNIW':
                    
                    dimu = prior_params['M'].shape[0]
                    dimX = prior_params['M'].shape[1]
                    
                    theta = {'invSigma':np.zeros([dimu,dimu,Kz,Ks]),'A':np.zeros([dimu,dimX,Kz,Ks])}
                    
                elif model['obsModel']['priorType']=='MNIW-N' or model['obsModel']['priorType']=='N-IW-N':
                    
                    dimu = prior_params['M'].shape[0]
                    dimX = prior_params['M'].shape[1]
                    
                    theta = {'invSigma':np.zeros([dimu,dimu,Kz,Ks]),'A':np.zeros([dimu,dimX,Kz,Ks]),'mu':np.zeros([dimu,Kz,Ks])}
                    
                elif model['obsModel']['priorType']=='ARD':
                    
                    dimu = prior_params['M'].shape[0]
                    dimX = prior_params['M'].shape[1]
                    
                    theta = {'invSigma':np.zeros([dimu,dimu,Kz,Ks]),'A':np.zeros([dimu,dimX,Kz,Ks]),'mu':np.zeros([dimu,Kz,Ks]),'ARDypers':np.zeros([dimX,Kz,Ks])}
               
                elif model['obsModel']['priorType']=='Afixed-IW-N':
                    
                    dimu = prior_params['A'].shape[0]
                    dimX = prior_params['A'].shape[1]
                    
                    theta = {'invSigma':np.zeros([dimu,dimu,Kz,Ks]),'A':np.kron(np.ones([1,1,Kz,Ks]),prior_params['A']),'mu':np.zeros([dimu,Kz,Ks])}
                    
                else
                    raise ValueError('no known prior type')
            
            
            Ustats = struct('card',zeros(Kz,Ks),'XX',zeros(dimX,dimX,Kz,Ks),'YX',zeros(dimu,dimX,Kz,Ks),'YY',zeros(dimu,dimu,Kz,Ks),'sumY',zeros(dimu,Kz,Ks),'sumX',zeros(dimX,Kz,Ks));
            
            if model['obsModel']['type']=='SLDS':
                
                model['obsModel']['r'] = 1
                
                if 'Kr' in settings.keys():
                    Kr = 1
                    model['HMMmodel']['params']['a_eta'] = 1
                    model['HMMmodel']['params']['b_eta'] = 1
                    print('Using single Gaussian measurement noise model')
                else
                    Kr = settings['Kr']
                    print('Using mixture of Gaussian measurement noise model')
                
                
                dimy = prior_params['C'].shape[0]
                
                    
                if model['obsModel']['y_priorType']=='IW':
                        theta['theta_r'] = {'invSigma',np.zeros([dimy,dimy,Kr])}

                elif model['obsModel']['y_priorType']=='NIW' or model['obsModel']['y_priorType']=='IW-N':
                        theta['theta_r'] = {'invSigma',np.zeros([dimy,dimy,Kr]),'mu':np.zeros([dimy,Kr])}
                else
                    raise ValueError('no known prior type for measurement noise')
                
                
                Ustats['Ustats_r'] = {'card':np.zeros([1,Kr]),'YY':np.zeros([dimy,dimy,Kr]),'sumY':np.zeros([dimy,Kr])}
                hyperparams['eta0'] = 0
                stateCounts['Nr'] = np.zeros([1,Kr])
                            
        
            
            for ii in range(0,len(data_struct))
                if 'X' not in data_struct[ii].keys() or np.size(data_struct[ii]['X'])==0:
                    
                    X,valid = makeDesignMatrix(data_struct[ii]['obs'],model['obsModel']['r'])
                    
                    data_struct[ii]['obs'] = data_struct[ii]['obs'][:,valid.ravel().nonzero()]
                    data_struct[ii]['X'] = X[:,valid.ravel().nonzero()];
                    if np.size(data_struct[ii]['blockSize'])==0:
                        data_struct[ii]['blockSize'] = np.ones([1,data_struct[ii]['obs']])
                
                    data_struct(ii).blockEnd = cumsum(data_struct(ii).blockSize);
                    if isfield(data_struct(ii),'true_labels')
                        data_struct(ii).true_labels = data_struct(ii).true_labels(find(valid));


    numObj = length(data_struct);

    stateCounts.N = zeros(Kz+1,Kz,numObj);
    stateCounts.Ns = zeros(Kz,Ks,numObj);

    hyperparams.gamma0 = 0;
    hyperparams.alpha0 = 0;
    hyperparams.kappa0 = 0;          
    hyperparams.sigma0 = 0;

    numSaves = settings.saveEvery/settings.storeEvery;
    S(1:numSaves) = struct('F',[],'config_log_likelihood',[],'theta',[],'dist_struct',[],'hyperparams',[],'stateSeq',[]);

theta Ustats stateCounts data_struct model S

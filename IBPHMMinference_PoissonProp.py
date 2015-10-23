def IBPHMMinference_PoissonProp(data_struct,model,settings,restart,init_params):
	'''
	Inputs:
	data_struct - dict of observations and any associated blockings
	prior_params - dict of hyperparameters for the prior on the model parameters
	hyperhyperparams - dict of hyperparameters on concentration parameters
	settings - dict of settings including truncation levels, number of Gibbs iterations, etc.

	Outputs:
	various statistics saved at each iteration as 'stats(Niter).mat' '''


	trial = settings['trial']
	if 'saveMin' in settings.keys():
	    settings['saveMin'] = 1

	Niter = settings['Niter']


try restart:
	if restart==1:
		n=settings['LastSave']

		if 'filename' in settings.keys():
			filename = settings['saveDir']+'/'+settings['filename']+'iter'+String(n)+'trial'+string(settings['trial'])    # create filename for current iteration
        else:
            filename = settings['saveDir']+'/IBPHMMstats'+'iter'+String(n)+'trial'+string(settings['trial'])   


		with open(filename,'wb') as f:
		
		 F = S[-1]['F']
		 theta,Ustats,stateCounts,data_struct,model,S = initializeStructs(F,model,data_struct,settings)
	        clear theta Ustats stateCounts S


	        load(filename)
	        
	        obsModel = model['obsModel']  # structure containing the observation model parameters
	        obsModelType = obsModel['type']   # type of emissions including Gaussian, multinomial, AR, and SLDS.
	        HMMhyperparams = model['HMMmodel']['params'] # hyperparameter structure for the HMM parameters
	        numObj = length(data_struct);
	        
	        dist_init_flag = 0;
	        theta_init_flag = 0;
	        hyperparams_init_flag = 0;

	        theta = S[-1]['theta']
	        dist_struct = S[-1]['dist_struct']
	        hyperparams = S[-1]['hyperparams']
	        
	        n_start = n + 1;
	        
	        if 'Kstar' in settings.keys():
	            Kstar = settings['Kstar']
	        else:
	            Kstar = numObj
	else:

		    #if settings['ploton']
		    #    H1 = figure;
		    #    H2 = figure; A2 = gca();
		    
		    
		    n_start = 1
		    
		    #Kstar = settings.Kstar;
		    
		    obsModel = model['obsModel']  # structure containing the observation model parameters
		    obsModelType = obsModel['type']   # type of emissions including Gaussian, multinomial, AR, and SLDS.
		    HMMhyperparams = model['HMMmodel']['params'] # hyperparameter structure for the HMM parameters
		    numObj = length(data_struct);
		    
		    if 'Kstar' in settings.keys():
		        Kstar = settings['Kstar']
		    else:
		        Kstar = numObj
		    
		    # Resample concentration parameters:
		    #hyperparams = sample_hyperparams_init(stateCounts,hyperparams,HMMhyperparams,HMMmodelType,resample_kappa);
		    hyperparams_init_flag = 0;
		    try init_params,var:
		        if 'hyperparams' in init_params.keys():        
		            hyperparams = init_params['hyperparams']
		            hyperparams_init_flag = 1
		    except NameError:
		    	pass

		    if not(hyperparams_init_flag):
		        hyperparams['alpha0'] = HMMhyperparams['a_alpha']/HMMhyperparams['b_alpha']
		        hyperparams['kappa0'] = HMMhyperparams['a_kappa']/HMMhyperparams['b_kappa']
		        hyperparams['sigma0'] = 1
		        hyperparams['gamma0'] = HMMhyperparams['a_gamma']/HMMhyperparams['b_gamma']
		    
		    
		    
		    F_init_flag = 0
		    try init_params,var:
		        if 'F' in init_params.keys(): 
		            F = init_params['F']
		            F_init_flag = 1
		    except NameError:
		    	pass    
		    
		    
		    if not(F_init_flag):
		    	if isfield(settings,'formZInit'):
		            for jj in range(0,length(data_struct)):
		                F[jj,unique(data_struct[jj]['z_init'])] = 1;
	       
	        	else
	            	F = np.ones([numObj,20])
     
        #F = sample_features_init(numObj,hyperparams.gamma0);
        
        #if settings['ploton']
            #imagesc(F,'Parent',A2); title(A2,['Featuer Matrix, Iter: ' num2str(n_start)]);
            #drawnow;
 
    
    
    
    # Build initial structures for parameters and sufficient statistics:
    theta,Ustats,stateCounts,data_struct,model,S = initializeStructs(F,model,data_struct,settings);
    
    # Sample the transition distributions pi_z, initial distribution
    # pi_init, emission weights pi_s, and global transition distribution beta
    # (only if HDP-HMM) from the priors on these distributions:
    dist_init_flag = 0
    try init_params,var:
        if dist_struct in init_params.keys():
            dist_struct = init_params.dist_struct
            dist_init_flag = 1
    expect NameError:
    	pass
    
    if not(dist_init_flag):
        #dist_struct = sample_dist(stateCounts,hyperparams,Kstar);
        dist_struct = sample_dist(stateCounts,hyperparams,Kstar);
    
    if 'formZInit' in settings.keys():
        Ustats_temp = Ustats
        stateSeq,INDS,stateCounts = sample_zs_init(data_struct,dist_struct,obsModelType)
        Ustats = update_Ustats(data_struct,INDS,stateCounts,obsModelType)
        if obsModelType=='SLDS':
            Ustats['Ustats_r'] = Ustats_temp['Ustats_r']
       
        numInitThetaSamples = 1
        print('Forming initial z using specified z_init or sampling from the prior using whatever fixed data is available')
    else:
        numInitThetaSamples = 1;
   
        
    # Sample emission params theta_{z,s}'s initially from prior (sometimes bad
    # choice):
    theta_init_flag = 0; 
    try init_params,var
        if 'theta' in init_params.keys():
            theta = init_params['theta']
            theta_init_flag = 1
    except NameError:
    	pass    
    
    
    if not(theta_init_flag):
        #theta = sample_theta(theta,Ustats,obsModel,Kstar);
        theta = sample_theta(theta,Ustats,obsModel,Kstar);
        for ii in range(0,numInitThetaSamples)
            #theta = sample_theta(theta,Ustats,obsModel,Kstar);
            theta = sample_theta(theta,Ustats,obsModel,0)
    
    if 'file' in settings['saveDir'].keys():
        os.mkdir(settings['saveDir'])
   
    # Save initial statistics and settings for this trial:
    if 'filename' in settings.keys():
        settings_filename = settings['saveDir']+'/'+settings['filename']+'_info4trial'+str(trial) #create filename for current iteration
        init_stats_filename = settings['saveDir']+'/'+settings['filename'+'initialStats_trial'+str(trial)  #create filename for current iteration
    else:
        settings_filename = settings['saveDir']+'/info4trial'+str(trial)  #create filename for current iteration
        init_stats_filename = settings['saveDir']+'/initialStats_trial'+str(trial) #create filename for current iteration
    
    #save(settings_filename,'data_struct','settings','model') % save current statistics
    #save(init_stats_filename,'dist_struct','theta','hyperparams') % save current statistics
    

        


except NameError:


return num_prop,num_accept,log_prob_tracker 
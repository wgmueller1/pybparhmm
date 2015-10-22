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

		    if !hyperparams_init_flag:
		        hyperparams['alpha0'] = HMMhyperparams['a_alpha']/HMMhyperparams['b_alpha']
		        hyperparams['kappa0'] = HMMhyperparams['a_kappa']/HMMhyperparams['b_kappa']
		        hyperparams['sigma0'] = 1
		        hyperparams['gamma0'] = HMMhyperparams['a_gamma']/HMMhyperparams['b_gamma']
		    
		    
		    
		    F_init_flag = 0
		    if exist('init_params','var')
		        if isfield(init_params,'F')
		            F = init_params.F;
		            F_init_flag = 1;
		        end
		    end
		    
		    if ~F_init_flag
        


except NameError:


return num_prop,num_accept,log_prob_tracker 
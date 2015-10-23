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
	    

	    fileObject = open(settings_filename,'wb') 
	    fileObject2 = open(init_stats_filename,'wb') 
	    pickle.dump((data_struct,settings,model),fileObject) # save current statistics
	    pickle.dump((dist_struct,theta,hyperparams),fileObject2) 
	   
	    total_length = 0
		length_ii = np.zeros([1,len(data_struct)])
		for ii in range(0,len(data_struct)):
		    length_ii(ii) = len(data_struct[ii]['true_labels'])
		    total_length = total_length + length_ii[ii]
		
		cummlength = np.cumsum(length_ii)
		z_tot = np.zeros([1,cummlength[-1]])
		true_labels_tot = np.eros([1,cummlength[-1]])
		true_labels_tot[0:length_ii[0]] = data_struct[0]['true_labels']
		for ii in range(1,len(data_struct)):
		    true_labels_tot[cummlength[ii-1]+1:cummlength[ii]] = data_struct[ii]['true_labels']
		

		try track_joint_prob,var:
			pass
		except NameError:
		    print('Not tracking joint probability')
		

	# Run the sampler
	log_prob_tracker = np.zeros([1,Niter])

	num_prop = np.zeros([Niter,Kstar])
	num_accept = np.zeros([Niter,Kstar])

	for n in range(n_start,Niter):
	   
	    #if ~F_init_flag
	        ##[F dist_struct theta config_log_likelihood] = sample_features(F,hyperparams.gamma0,data_struct,dist_struct,theta,obsModel,Kstar);
	        #[F dist_struct theta config_log_likelihood] = sample_features(F,hyperparams.gamma0,data_struct,dist_struct,theta,obsModel);
	    F,dist_struct,theta,config_log_likelihood,num_prop[n,:],num_accept[n,:]= sample_features_PoissonProp(F,hyperparams.gamma0,data_struct,dist_struct,theta,obsModel,hyperparams,Kstar)
	    
	    
	    # Sample z and s sequences given data, transition distributions,
	    # HMM-state-specific mixture weights, and emission parameters:
	    # Block sample z_{1:T}|y_{1:T}
	    stateSeq,INDS,stateCounts, = sample_zs_old(data_struct,dist_struct,F,theta,obsModelType)
	    # Create sufficient statistics:
	    Ustats = update_Ustats(data_struct,INDS,stateCounts,obsModelType)
	 
	    # Sample the transition distributions pi_z, initial distribution
	    # pi_init, emission weights pi_s, and avg transition distribution beta
	    # (only if HDP-HMM):
	    #
	    if not(dist_init_flag):
	        #dist_struct = sample_dist(stateCounts,hyperparams,Kstar);
	        dist_struct = sample_dist(stateCounts,hyperparams,Kstar);
	    
	    
	    # Sample theta_{z,s}'s conditioned on z and s sequences and data suff.
	    # stats. Ustats:
	    if not(theta_init_flag):
	        #theta = sample_theta(theta,Ustats,obsModel,Kstar);
	        theta = sample_theta(theta,Ustats,obsModel,Kstar)
	    
	    hyperparams = sample_IBPparam(F,hyperparams,HMMhyperparams)
	    
	    # Resample concentration parameters:
	    if not(hyperparams_init_flag):
	        hyperparams = sample_distparams(F,dist_struct,hyperparams,HMMhyperparams,50)
	    

	    # Build and save stats structure:
	    #S = store_stats(S,n,settings,F,config_log_likelihood,stateSeq,dist_struct,theta,hyperparams)
	    
	    # Plot stats:
	    if 'true_labels' in data_struct.keys() & settings['ploton']:
	                
	        if remainder(n,settings['plotEvery']==0:
	                        
	            F_used = np.zeros([F.shape])
	            Nsets = len(data_struct)
	            sub_x = floor(sqrt(Nsets))
	            sub_y = ceil(Nsets/sub_x)
	            
	            z_tot[0:length_ii[0]] = stateSeq[0]['z']            
	            for ii range(1,Nsets):
	                z_tot[cummlength[ii-1]+1:cummlength[ii]] = stateSeq[ii]['z']
	            
	            
	            relabeled_z,Hamm,assignment,relabeled_true_labels = mapSequence2Truth(true_labels_tot,z_tot)
	            
	            F_used[1,unique(stateSeq[0]['z'])] = 1
	            #A1 = subplot(sub_x,sub_y,1,'Parent',H1);
	            #imagesc([relabeled_z(1:cummlength(1)); relabeled_true_labels(1:cummlength(1))],'Parent',A1,[1 max(union(relabeled_z,relabeled_true_labels))]); colorbar('peer',A1); title(A1,['Iter: ' num2str(n)]);
	            for ii in range(1,Nsets)
	                 F_used[ii,unique(stateSeq[ii]['z'])] = 1
	                #A1 = subplot(sub_x,sub_y,ii,'Parent',H1);
	                #imagesc([relabeled_z(cummlength(ii-1)+1:cummlength(ii)); relabeled_true_labels(cummlength(ii-1)+1:cummlength(ii))],'Parent',A1,[1 max(union(relabeled_z,relabeled_true_labels))]); colorbar('peer',A1);  title(A1,['Iter: ' num2str(n)]); 
	           
	            #plt.show()
	            
	            #imagesc(F+F_used,'Parent',A2); title(A2,['Featuer Matrix, Iter: ' num2str(n)]);
	            #plt.show()
	            
	            #if isfield(settings,'plotpause') && settings.plotpause
	            #    if isnan(settings.plotpause), waitforbuttonpress; else pause(settings.plotpause); end


	#fname = [settings.saveDir '/num_accept_prop_trial' num2str(trial)];

	#save(fname,'num_accept','num_prop')

	except NameError:
		pass


	return num_prop,num_accept,log_prob_tracker 
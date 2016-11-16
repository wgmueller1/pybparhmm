def sample_features_PoissonProp(F_prev,gamma0,data_struct,dist_struct,theta,obsModel,hyperparams,Kstar):
  num_accept = np.zeros([1,Kstar])
  num_prop = np.zeros([1,Kstar])

  obsModelType = obsModel['type']
  priorType = obsModel['priorType']

  numObj,Kz_prev = F_prev.shape
  Kz_max = Kz_prev + Kstar
  F = np.zeros([numObj,Kz_max])
  F[:,0:Kz_prev] = F_prev
  F = np.where(F > 0)
  featureCounts = np.sum(F,axis=0)
  stored_log_likelihood = np.zeros([1,numObj])
  Ks = dist_struct[0]['pi_s'].shape[1]

  log_likelihood_ii_kk = np.array([0,0])

  seq_order = np.random.permutation(numObj)

  feature_inds = range(0,Kz_max)
  config_log_likelihood = 0
  for ii in range(seq_order):
	    # Reset vector indicating the previous set of unique features to object i:
      unique_features_ii = np.zeros([1,Kz_max])
      unique_features_ii = (unique_features_ii > 0)
	    
	    # Compute likelihood under all possible parameters (including ones we
	    # have not yet seen in the data):
      log_likelihood = compute_likelihood_unnorm(data_struct[ii],theta,obsModelType,range(0,Kz_max),Kz_max,Ks)
      dimu = data_struct[ii]['obs'].shape[0]
	    
	    # Compute current likelihood of the current feature assignments:
      if np.sum(F[ii,:]) == 0:
        stored_log_likelihood[ii] = -np.inf
      else:
	        
          pi_init = dist_struct[ii]['pi_init'][F[ii,:]]
          pi_init = pi_init/np.sum(pi_init)
          pi_z = dist_struct[ii]['pi_z'][F[ii,:],F[ii,:]]
          pi_z = pi_z/matlib.repmat(np.sum(pi_z,axis=1),[1,pi_z.shape[1]])
          pi_s = dist_struct[ii]['pi_s'][F[ii,:]]
          pi_s = pi_s/matlib.repmat(np.sum(pi_s,axis=1),[1,pi_s.shape[1]])
	        
	        # Pass messages forward to integrate over the mode/state sequence:
          log_likelihood_ii = log_likelihood[F[ii,:],:,:]
          log_normalizer_ii = np.max(np.max(log_likelihood_ii,axis=0),axis=1)
          log_likelihood_ii = log_likelihood_ii - log_normalizer_ii[np.ones([np.sum(F[ii,:]),1]),np.ones([Ks,1]),:]
          likelihood_ii = np.exp(log_likelihood_ii)
          log_normalizer_ii = log_normalizer_ii - (dimu/2)*log(2*pi);
	        
          fwd_msg,neglog_c = forward_message_vec(likelihood_ii,log_normalizer_ii,data_struct[ii]['blockEnd'],pi_z,pi_s,pi_init)
	        
          if np.isnan(np.sum(neglog_c)):
              stored_log_likelihood(ii) = -np.inf
          else:
            stored_log_likelihood(ii) = np.sum(neglog_c) #observation_likelihood(F(ii,:),data_struct(ii),obsModelType,dist_struct(ii),theta);
    # For each of the currently instantiated features (this vector will	
    # change after sampling each object ii):
      for kk in feature_inds[featureCounts>0]:
        
        # Store previous feature value:
        Fik_prev = F[ii,kk]
        # Remove object i's count from the kth feature count:
        featureCounts[kk] = featureCounts[kk]-F[ii,kk]
        
        # If other objects are using this feature:
        if featureCounts(kk)>0:
                      
            # Grab out previous likelihood of data under this feature
            # assignment:
            log_likelihood_ii_kk[Fik_prev+1] = stored_log_likelihood[ii]
            
            # Try out other value for f_{ik}:
            F(ii,kk) = np.logical_not(Fik_prev) 
            
            if np.sum(F[ii,:]) == 0:
                log_likelihood_ii_kk[np.logicial_not(Fik_prev)+1] = -np.inf
            else:             
                pi_init = dist_struct[ii]['pi_init'][F[ii,:]]
                pi_init = pi_init/np.sum(pi_init)
                pi_z = dist_struct[ii]['pi_z'][F[ii,:],F[ii,:]]
                pi_z = pi_z/matlib.repmat(np.sum(pi_z,axis=1),[1,pi_z.shape[1]])
                pi_s = dist_struct[ii]['pi_s'][F[ii,:]]
                pi_s = pi_s/matlib.repmat(np.sum(pi_s,axis=1),[1,pi_s.shape[1]])
                
                # Pass messages forward to integrate over the mode/state sequence:
                log_likelihood_ii = log_likelihood[F[ii,:],:,:]
                log_normalizer_ii = np.max(np.max(log_likelihood_ii,axis=0),axis=1)
                log_likelihood_ii = log_likelihood_ii - log_normalizer_ii[np.ones([np.sum(F[ii,:]),1]),np.ones([Ks,1]),:]
                likelihood_ii = np.exp(log_likelihood_ii)
                log_normalizer_ii = log_normalizer_ii - (dimu/2)*log(2*pi)
                
                fwd_msg,neglog_c = forward_message_vec(likelihood_ii,log_normalizer_ii,data_struct[ii]['blockEnd'],pi_z,pi_s,pi_init)
                
                if np.isnan(np.sum(neglog_c)):
                    log_likelihood_ii_kk[np.logical_not(Fik_prev)+1] = -np.inf
                else:
                    log_likelihood_ii_kk(np.logical_not(Fik_prev)+1) = np.sum(neglog_c) #observation_likelihood(F(ii,:),data_struct(ii),obsModelType,dist_struct(ii),theta);
                
            
            # Compute accept-reject ratio:
            log_rho_star = log(numObj - featureCounts[kk]) + log_likelihood_ii_kk[0]-log(featureCounts[kk]) \
            - log_likelihood_ii_kk[1]
            rho = np.exp(np.sign(Fik_prev-0.5)*log_rho_star)
            
            # Sample new feature value:
            if isnan(rho):
                F[ii,kk]=0
            else:
                
                if rho>1:
                    F[ii,kk] = np.logical_not(Fik_prev)
                else:
                    sample_set = np.array([Fik_prev,np.logical_not(Fik_prev)])
                    ind = (random.random()>(1-rho))
                    F[ii,kk] = sample_set[ind]

            
            # Store likelihood of current assignment:
            stored_log_likelihood[ii] = log_likelihood_ii_kk[F[ii,kk]+1]
        
            # Add new assignment of f_{ik} to feature count of kth feature:
            featureCounts[kk] = featureCounts[kk]+F[ii,kk]
            
        else:     
            # If kth feature is specific to object i, place it in the
            # indicator vector:
            unique_features_ii[kk] = 1
        
      #deal with unique features
    unique_feature_inds = feature_inds[unique_features_ii]
      
      # Sample from Poisson proposal;
    num_new_unique_features = poissrnd(gamma0/numObj)
      
    num_prop[num_new_unique_features+1] = num_prop[num_new_unique_features+1]+1
      
    f_ii = F[ii,:]
    f_ii[unique_feature_inds] = 0
    f_ii[Kz_prev+1:Kz_prev+num_new_unique_features] = 1
     
      #Grab likelihood under the previous assignment:
    log_likelihood_ii_kk(1) = stored_log_likelihood(ii);
     
     # Compute likelihood under the proposed change:
    if np.sum(f_ii) == 0:
         log_likelihood_ii_kk(2) = -np.inf
    else:
         
         if Kz_prev+num_new_unique_features>Kz_max:
             
             Kz_extra = Kstar + num_new_unique_features
             
             # Expand transition distributions:
             for jj in range(0,numObj):
                 dist_struct[jj]['pi_init'][Kz_max+1:Kz_max+Kz_extra] = randgamma(hyperparams.alpha0*ones(1,Kz_extra))
                 dist_struct[jj]['pi_z'][:,Kz_max+1:Kz_max+Kz_extra]= randgamma(hyperparams.alpha0*ones(Kz_max,Kz_extra))
                 dist_struct[jj]['pi_z'][Kz_max+1:Kz_max+Kz_extra,1:Kz_max] = randgamma(hyperparams.alpha0*ones(Kz_extra,Kz_max))
                 dist_struct[jj]['pi_z'][Kz_max+1:Kz_max+Kz_extra,Kz_max+1:Kz_max+Kz_extra] = randgamma(hyperparams.alpha0*ones(Kz_extra)\
                     + hyperparams['kappa0']*np.eye(Kz_extra))
                 dist_struct[jj]['pi_s'][Kz_max+1:Kz_max+Kz_extra,:] = 1
       
             
             # Expand theta:
             theta = sample_theta_extra(theta,obsModel,Kz_extra)
             
             Kz_max = Kz_max + Kz_extra;
             
             # Expand F:
             F_prev = F
             F = np.zeros([numObj,Kz_max])
             F = F > 0
             F[:,1:Kz_max-Kz_extra] = F_prev
             f_ii_prev = f_ii
             f_ii = np.zeros(1,Kz_max)
             f_ii = f_ii>0
             f_ii[0:length(f_ii_prev)] = f_ii_prev
             
             log_likelihood = compute_likelihood_unnorm(data_struct[ii],theta,obsModelType,range(0,Kz_max),Kz_max,Ks)
             
             print('adding more parameters')

    pi_init = dist_struct[ii]['pi_init'][f_ii]
    pi_init = pi_init/sum(pi_init);
    pi_z = dist_struct(ii).pi_z(f_ii,f_ii);
    pi_z = pi_z/matlab.repmat(sum(pi_z,2),[1,size(pi_z,2)]);
    pi_s = dist_struct(ii).pi_s(f_ii);
    pi_s = pi_s/matlab.repmat(sum(pi_s,2),[1,size(pi_s,2)]);
         
         # Pass messages forward to integrate over the mode/state sequence:
         log_likelihood_ii = log_likelihood(f_ii,:,:);
         log_normalizer_ii = max(max(log_likelihood_ii,[],1),[],2);
         log_likelihood_ii = log_likelihood_ii - log_normalizer_ii(ones(sum(f_ii),1),ones(Ks,1),:);
         likelihood_ii = exp(log_likelihood_ii);
         log_normalizer_ii = log_normalizer_ii - (dimu/2)*log(2*pi);
         
         [fwd_msg neglog_c] = forward_message_vec(likelihood_ii,log_normalizer_ii,data_struct(ii).blockEnd,pi_z,pi_s,pi_init);
         
         if isnan(sum(neglog_c))
             log_likelihood_ii_kk(2) = -inf;
         else
             log_likelihood_ii_kk(2) = sum(neglog_c); %observation_likelihood(F(ii,:),data_struct(ii),obsModelType,dist_struct(ii),theta);
         end
     end
     
     # Compute accept-reject ratio:
     log_rho_star = (log_likelihood_ii_kk[1] - log_likelihood_ii_kk[0]
     rho = np.exp(log_rho_star)
     
     # Sample new feature value:
     # doublecheck
     if np.isnan(rho):
         raise ValueError('NaN rho')
     else:
         if rho>1:
             F[ii,:] = f_ii
             ind = 0
         else:
             ind = (random.random()>(1-rho)
             F[ii,:] = (1-ind)*F[ii,:] + (ind-0)*f_ii
         
         
         num_accept[num_new_unique_features] = num_accept[num_new_unique_features]+ind
         

     
     #display(num2str((ind-0)*['accept proposal'] + (1-ind)*['reject proposal']))
     #
     #    if (ind==1) && (transition_case>1)
     #        removed_features(unique_feature_inds(death_ind)) = 1;
     #    end
     
  #    if log_likelihood_ii_kk(ind+1)<stored_log_likelihood(ii)
  #        display('accepted lower likelihood move')
  #    else
  #        display('moved to higher likelihood')
  #    end
     
     stored_log_likelihood[ii] = log_likelihood_ii_kk[ind+1]
     
     config_log_likelihood = config_log_likelihood + stored_log_likelihood[ii]
     
     featureCounts = np.sum(F,axis=0)
     
     used_features = np.where(featureCounts>0)
     Kz_prev = used_features[-1]
     

   F,dist_struct,theta = reallocate_states(F,dist_struct,theta,priorType)
   return F,display,theta
    



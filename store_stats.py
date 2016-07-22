import pickle

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def store_stats(S,n,settings,F_n,config_log_likelihood_n,stateSeq_n,dist_struct_n,theta_n,hyperparams_n):

    if n%settings['storeEvery']==0 & n>=settings['saveMin']:
        
        storeCount = n % settings['saveEvery']
        try: 
            storeCount
        except NameError:
            storeCount = settings.saveEvery
      
        
        S['storeCount']['stateSeq'] = stateSeq_n
        S['storeCount']['dist_struct'] = dist_struct_n
        S['storeCount']['F'] = F_n
        S['storeCount']['config_log_likelihood'] = config_log_likelihood_n
        S['storeCount']['theta'] = theta_n
        S['storeCount']['hyperparams'] = hyperparams_n  

        
    if n% settings.saveEvery==0:

        # Save stats:
        if 'filename' in settings.keys():
            filename = settings['saveDir']+'/'+settings['filename']+'iter'+str(n)+'trial'+str(settings['trial'])    # create filename for current iteration
        else:
            filename = settings['saveDir']+'/IBPHMMstats'+'iter'+str(n)+'trial'+str(settings['trial'])  # create filename for current iteration
       

        save_obj(S,filename) # save current statistics

        print('Iteration: '+ str(n))
     



def update_Ustats(data_struct,INDS,stateCounts,obsModelType):

    Ns = stateCounts['Ns']

    Kz = np.shape(Ns)[0]
    Ks = np.shape(Ns)[1]

    if obsModelType=='Gaussian':
            
            dimu = np.shape(data_struct[0]['obs'])[0]

            store_YY = np.zeros((dimu,dimu,Kz,Ks))
            store_sumY = np.zeros((dimu,Kz,Ks))
            store_card = np.zeros((Kz,Ks))
            
            for ii in range(0,len(data_struct)):
                
                unique_z = np.nonzero(np.sum(Ns[:,:,ii],axis=1)).T

                u = data_struct[ii]['obs']
                
                for kz in unique_z:
                    unique_s_for_z = np.nonzero(Ns[kz,:,ii])
                    for ks in unique_s_for_z:
                        obsInd = INDS[ii]['obsIndzs'][kz,ks]['inds'][0:INDS[ii]][obsIndzs[kz,ks]['tot']]
                        store_YY[:,:,kz,ks] = store_YY[:,:,kz,ks] + u[:,obsInd]*u[:,obsInd].T
                        store_sumY[:,kz,ks] = store_sumY[:,kz,ks] + np.sum(u[:,obsInd],axis=1)
                store_card = store_card + Ns[:,:,ii]            
          
            Ustats.card = store_card
            Ustats.YY = store_YY
            Ustats.sumY = store_sumY
    elif obsModelType in ['AR','SLDS']:

            dimu = np.shape(data_struct[1]['obs'][0])
            dimX = np.shape(data_struct[1]['X'][0])

            store_XX = np.zeros((dimX,dimX,Kz,Ks))
            store_YX = np.zeros((dimu,dimX,Kz,Ks))
            store_YY = np.zeros((dimu,dimu,Kz,Ks))
            store_sumY = np.zeros((dimu,Kz,Ks))
            store_sumX = np.zeros((dimX,Kz,Ks))
            store_card = np.zeros((Kz,Ks))

            for ii in range(0,len(data_struct)):
                
                unique_z = np.nonzero(np.sum(Ns[:,:,ii],axis=1)).T

                u = data_struct[ii]['obs']
                X = data_struct[ii]['X']

                for kz in unique_z:
                    unique_s_for_z = np.nonzero(Ns[kz,:,ii])
                    for ks in unique_s_for_z:
                        obsInd = INDS[ii][obsIndzs[kz,ks]['inds'][0:INDS[ii]]['obsIndzs'][kz,ks]['tot']]
                        store_XX[:,:,kz,ks] = store_XX[:,:,kz,ks] + X[:,obsInd]*X[:,obsInd].T
                        store_YX[:,:,kz,ks] = store_YX[:,:,kz,ks] + u[:,obsInd]*X[:,obsInd].T
                        store_YY[:,:,kz,ks] = store_YY[:,:,kz,ks] + u[:,obsInd]*u[:,obsInd].T
                        store_sumY[:,kz,ks] = store_sumY[:,kz,ks] + np.sum(u[:,obsInd],axis=1)
                        store_sumX[:,kz,ks] = store_sumX[:,kz,ks] + np.sum(X[:,obsInd],axis=1)

                store_card = store_card + Ns[:,:,ii]
                

                   
            Ustats['card'] = store_card
            Ustats['XX'] = store_XX
            Ustats['YX'] = store_YX
            Ustats['YY'] = store_YY
            Ustats['sumY'] = store_sumY
            Ustats['sumX'] = store_sumX
            
            if obsModelType=='SLDS' and 'Nr' in stateCounts.keys():  # Don't update if just using z_init
                
                Nr = stateCounts['Nr']
                Kr = len(Nr)
                unique_r = np.nonzero(Nr)
                
                dimy = np.shape(data_struct[0]['tildeY'])[0]
                
                store_tildeYtildeY = np.zeros((dimy,dimy,Kr))
                store_sumtildeY = np.zeros((dimy,Kr))
                store_card = np.zeros((1,Kr))
                
                for ii in range(0,len(data_struct)):
                    
                    tildeY = data_struct[ii]['tildeY']
                    for kr in unique_r:
                        obsInd_r = INDS[ii]['obsIndr'][kr['inds'][0:INDS[ii]['obsIndr'][kr]['tot']]]
                        store_tildeYtildeY[:,:,kr] = store_tildeYtildeY[:,:,kr] + tildeY[:,obsInd_r]*tildeY[:,obsInd_r].T
                        store_sumtildeY[:,kr] = store_sumtildeY[:,kr] + np.sum(tildeY[:,obsInd_r],axis=1)
                    
                    store_card = store_card + Nr[ii,:]
                    
            
                
                Ustats.Ustats_r['YY'] = store_tildeYtildeY
                Ustats.Ustats_r['sumY'] = store_sumtildeY
                Ustats.Ustats_r['card'] = store_card

            
    elif obsModelType=='Multinomial':

            numVocab = data_struct[0]['numVocab']
            
            store_counts = np.zeros((numVocab,Kz,Ks))

            for ii in range(0,len(data_struct)):
                u = data_struct[ii]['obs']

                unique_z = np.nonzero(np.sum(Ns[:,:,ii],axis=1)).T

                for kz in unique_z:
                    unique_s_for_z = np.nonzero(Ns[kz,:,ii])
                    for ks in unique_s_for_z:
                        obsInd = INDS[ii]['obsIndzs'][kz,ks]['inds'][0:INDS[ii]]['obsIndzs'][kz,ks]['tot']
                        store_counts[:,kz,ks] = store_counts[:,kz,ks] + np.histogram(u['obsInd'],range(0,numVocab)).T
         
            
            Ustats.card = store_counts

    return Ustats
import numpy as np

def sample_zs_init(data_struct,dist_struct,obsModelType):

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define and initialize parameters %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Kz = np.shape(dist_struct[1]['pi_z'][1]
Ks = np.shape(dist_struct[1]['pi_s'][1]

# Preallocate INDS
for ii in range(0,len(data_struct)):
  T = len(data_struct[ii]['blockSize']
  INDS[ii][obsIndzs[0:Kz,0:Ks] = dict('inds',np.sparse(1,T),'tot',0)



for ii in range(0,len(data_struct)):
    
    N_init = np.zeros((Kz+1,Kz))
    Ns_init = np.zeros((Kz,Ks))
    
    if 'z_init' in data_struct[ii].keys():
        z_init_ii = data_struct[ii]['z_init']
    else:
        z_init_ii = data_struct[ii]['true_labels']

    
    stateSeq(ii).z,stateSeq(ii).s,totSeq,indSeq,N(:,:,ii),Ns(:,:,ii) = setZtoFixedSeq(data_struct[ii],dist_struct[ii],N_init,Ns_init,z_init_ii,0)
    
    for jj in range(0,Kz):
        for kk in range(0,Ks):
            INDS[ii][obsIndzs[jj,kk]]['tot']  = totSeq[jj,kk]
            INDS[ii][obsIndzs[jj,kk]['inds'] = scipy.sparse(indSeq[:,jj,kk].T)


stateCounts['N'] = N
stateCounts['Ns'] = Ns

return (stateSeq,INDS,stateCounts)


def sampleZfromPrior(data_struct,dist_struct,N,Ns):

    # Define parameters:
    pi_z = dist_struct['pi_z']
    pi_s = dist_struct['pi_s']
    pi_init = dist_struct['pi_init']

    Kz = np.shape(pi_z)[1]
    Ks = np.shape(pi_s)[1]

    T = len(data_struct['blockSize'])
    blockSize = data_struct['blockSize']
    blockEnd = data_struct['blockEnd']

    # Initialize state and sub-state sequences:
    z = np. zeros((1,T))
    s = np.zeros((1,np.sum(blockSize)))

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Sample the state and sub-state sequences %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Sample (z(1),{s(1,1)...s(1,N1)}).  We first sample z(1) given the
    # observations u(1,1)...u(1,N1) having marginalized over the associated s's
    # and then sample s(1,1)...s(1,N1) given z(1) and the observations.

    totSeq = np.zeros((Kz,Ks))
    indSeq = np.zeros((T,Kz,Ks))

    # for t=1:T
    #     % Sample z(t):
    #     if (t == 1)
    #         Pz = pi_init';
    #         obsInd = [1:blockEnd(1)];
    #     else
    #         Pz = pi_z(z(t-1),:)';
    #         obsInd = [blockEnd(t-1)+1:blockEnd(t)];
    #     end
    #     Pz   = cumsum(Pz);
    #     z(t) = 1 + sum(Pz(end)*rand(1) > Pz);
        
    #     % Add state to counts matrix:
    #     if (t > 1)
    #         N(z(t-1),z(t)) = N(z(t-1),z(t)) + 1;
    #     else
    #         N(Kz+1,z(t)) = N(Kz+1,z(t)) + 1;  % Store initial point in "root" restaurant Kz+1
    #     end
        
    #     % Sample s(t,1)...s(t,Nt) and store sufficient stats:
    #     for k=1:blockSize(t)
    #         % Sample s(t,k):
    #         if Ks > 1
    #             Ps = pi_s(z(t),:);
    #             Ps = cumsum(Ps);
    #             s(obsInd(k)) = 1 + sum(Ps(end)*rand(1) > Ps);
    #         else
    #             s(obsInd(k)) = 1;
    #         end
            
    #         % Add s(t,k) to count matrix and observation statistics:
    #         Ns(z(t),s(obsInd(k))) = Ns(z(t),s(obsInd(k))) + 1;
    #         totSeq(z(t),s(obsInd(k))) = totSeq(z(t),s(obsInd(k))) + 1;
    #         indSeq(totSeq(z(t),s(obsInd(k))),z(t),s(obsInd(k))) = obsInd(k);
    #     end
    # end

    # return (z,s,totSeq,indSeq,N,Ns)  


    # function [z s totSeq indSeq N Ns] = setZtoFixedSeq(data_struct,dist_struct,N,Ns,z_fixed,sampleS)
        
    #     % Define parameters:
    #     pi_z = dist_struct.pi_z;
    #     pi_s = dist_struct.pi_s;
    #     pi_init = dist_struct.pi_init;
        
    #     Kz = size(pi_z,2);
    #     Ks = size(pi_s,2);
        
    #     T = length(data_struct.blockSize);
    #     blockSize = data_struct.blockSize;
    #     blockEnd = data_struct.blockEnd;
        
    #     totSeq = zeros(Kz,Ks);
    #     indSeq = zeros(T,Kz,Ks);
        
    #     % Initialize state and sub-state sequences:
    #     z = z_fixed;
    #     if sampleS
    #         for t=1:T
    #             % Sample z(t):
    #             if (t == 1)
    #                 obsInd = [1:blockEnd(1)];
    #             else
    #                 obsInd = [blockEnd(t-1)+1:blockEnd(t)];
    #             end
                
    #             % Sample s(t,1)...s(t,Nt) and store sufficient stats:
    #             for k=1:blockSize(t)
    #                 % Sample s(t,k):
    #                 if Ks > 1
    #                     Ps = pi_s(z(t),:);
    #                     Ps = cumsum(Ps);
    #                     s(obsInd(k)) = 1 + sum(Ps(end)*rand(1) > Ps);
    #                 else
    #                     s(obsInd(k)) = 1;
    #                 end
    #             end
    #         end
    #     else
    #         s = ones(1,sum(blockSize));
    #     end
        
        
    #     for t=1:T
    #         % Sample z(t):
    #         if (t == 1)
    #             obsInd = [1:blockEnd(1)];
    #         else
    #             obsInd = [blockEnd(t-1)+1:blockEnd(t)];
    #         end
            
    #         % Add state to counts matrix:
    #         if (t > 1)
    #             N(z(t-1),z(t)) = N(z(t-1),z(t)) + 1;
    #         else
    #             N(Kz+1,z(t)) = N(Kz+1,z(t)) + 1;  % Store initial point in "root" restaurant Kz+1
    #         end
            
    #         % Sample s(t,1)...s(t,Nt) and store sufficient stats:
    #         for k=1:blockSize(t)
                
    #             % Add s(t,k) to count matrix and observation statistics:
    #             Ns(z(t),s(obsInd(k))) = Ns(z(t),s(obsInd(k))) + 1;
    #             totSeq(z(t),s(obsInd(k))) = totSeq(z(t),s(obsInd(k))) + 1;
    #             indSeq(totSeq(z(t),s(obsInd(k))),z(t),s(obsInd(k))) = obsInd(k);
    #         end
    #     end
        
    #     
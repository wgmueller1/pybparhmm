def forward_message_vec(likelihood,loglike_normalizer,blockEnd,pi_z,pi_s,pi_init):
	#Allocate storage space
	Kz = pi_z.shape[1]
	Ks = pi_s.shape[1]
	T  = len(blockEnd)

	fwd_msg  = np.ones([Kz,T])
	neglog_c = np.zeros([1,T])
	marg_like = np.sum(likelihood*pi_s[:,:,np.ones([1,1,blockEnd[-1]])],axis=1).reshape(likelihood.shape[0],likelihood.shape[2])
	
	# If necessary, combine likelihoods within blocks, avoiding underflow
	if T < blockEnd(end):
	  marg_like = log(marg_like+eps);

	  block_like = np.zeros([Kz,T]);
	  block_like[:,1] = np.sum(marg_like[:,1:blockEnd[0]],axis=1)
	  # Initialize normalization constant to be that due to the likelihood:
	  neglog_c[0] = np.sum(loglike_normalizer[1:blockEnd[0]])
	  for tt in range(1,T):
	    block_like[:,tt] = np.sum(marg_like[:,blockEnd[tt-1]+1:blockEnd[tt]],axis=1)
	    neglog_c[tt] = np.sum(loglike_normalizer[blockEnd[tt-1]+1:blockEnd[tt]])
	 

	  block_norm = np.max(block_like,axis=0)
	  block_like = exp(block_like - block_norm[np.ones([Kz,1]),:])
	  # Add on the normalization constant used after marginalizing the s_t's:
	  neglog_c = neglog_c + block_norm;
	else:
	   
	  block_like = marg_like;
	  # If there is no blocking, the normalization is simply due to the
	  # likelihood computation:
	  d1=loglike_normalizer.shape[1]
	  d2=loglike_normalizer.shape[2]
	  neglog_c = loglike_normalizer.reshape(d1,d2).T

	return fwd_msg, neglog_c, block_like


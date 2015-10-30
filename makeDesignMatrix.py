def makeDesignMatrix(Y,order):
	'''function [X,valid] = makeDesignMatrix(Y,order)
		# Y = d x T
		# X = order*d x T
		#
		# X = [0 Y(:,1:(end-1); 0 0 Y(:,1:(end-2)); 0 0 0 Y...etc]
		#
		# valid(t) = 1 for all X(:,t) where zeros were not inserted
		#
		function [X,valid]= makeDesignMatrix(Y,order)'''

	d = Y.shape[0]
	T = Y.shape[1]

	X = np.zeros([order*d,T])

	for lag in range(0,order):
	  ii   = d*(lag-1)+1
	  indx = range(ii,(ii+d))
	  X[indx, :] = [np.zeros([d,min(lag,T)]),Y[:,0:(T-min(lag,T))]]


	if nargout > 1:
	  valid = np.ones([1,T])
	  valid[0,order] = 0

	return X,valid

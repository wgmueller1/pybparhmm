def randdirichlet(a):
	# RANDDIRICHLET   Sample from Dirichlet distribution
	#
	# X = RANDDIRICHLET(A) returns a matrix, the same size as A, where X(:,j)
	# is sampled from a Dirichlet(A(:,j)) distribution.

	gamma=np.vectorize(np.random.gamma)
	x=gamma(a)
	return x

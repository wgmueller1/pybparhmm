def sample_IBPparam(F,hyperparams,hyperhyperparams):

	a_gamma=hyperhyperparams['a_gamma']
	b_gamma=hyperhyperparams['b_gamma']
	harmonic = hyperhyperparams['harmonic']

	Kplus = np.sum(np.sum(F,axis=0)>0)
	#doublecheck
	randgamma=np.vectorize(np.random.gamma)
	gamma0 = randgamma(a_gamma + Kplus) / (b_gamma + harmonic)

	hyperparams['gamma0'] = gamma0
	return hyperparams
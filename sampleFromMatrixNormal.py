#function S = sampleFromMatrixNormal(M,V,K,nSamples=1)
def sampleFromMatrixNormal(M,sqrtV,sqrtinvK,nSamples):

	try:
		nSamples,var
	except NameError:
		nSamples = 1

	[mu,sqrtsigma] = matrixNormalToNormal(M,sqrtV,sqrtinvK)

	S = mu + sqrtsigma.T*randn(len(mu),1)
	S = np.reshape(S,np.shape(M))
	return S
def matrixNormalToNormal(M,sqrtV,sqrtinvK):
	'''converts the parameters for a matrix normal A ~ MN(M,V,K) 
      into a  multivariate normal  A(:) ~ N(mu,sigma)'''

      mu = M[:]
      sqrtsigma = np.kron(sqrtinvK,sqrtV)

      return mu,sqrtsigma
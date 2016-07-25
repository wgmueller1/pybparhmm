def transformDistStruct(dist_struct,feature_vec):

	Kz = len(feature_vec)
	pi_z = dist_struct['pi_z']*np.repmat(feature_vec,[Kz,1])
	pi_z = pi_z/np.repmat(np.sum(pi_z,axis=1),[1,Kz])
	pi_init = dist_struct['pi_init']*feature_vec
	pi_init = pi_init/np.sum(pi_init)

	return (pi_z,pi_init)
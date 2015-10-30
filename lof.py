def lof(F):
	Kz = F.shape[1]
	val = np.zeros([1,Kz])

	for kk in range(0,Kz):
	    col_kk = str(F[:,kk].T)
	    val[kk] = int(col_kk,2)

	val_sort= val[::-1].sort()
	sort_ind = val[::-1],argsort()
	F_sorted = F[:,sort_ind]

	return F_sorted,sort_ind
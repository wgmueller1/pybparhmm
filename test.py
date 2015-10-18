from stats_util import *
import numpy as np
from scipy import random, linalg
p = 10 
A = random.rand(p,p)
B = np.dot(A,A.transpose())
print 'random positive semi-define matrix for today is', B

print sample_invwishart(B,15)
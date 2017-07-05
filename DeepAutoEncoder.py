from Gaussian_RBM import Gaussian_RBM
from Bernoulli_RBM import Bernoulli_RBM
import numpy as np

import matplotlib.pyplot as plt

from LoadDataSet import documents

from scipy import spatial

from FeatureExtraction_Local import listOfArrays

randomNumberGenerator = np.random.RandomState(2222)

gRBM = Gaussian_RBM(listOfArrays[0][0] , 50 ,70 ,randomNumberGenerator,0.3)

gRBM.updations()
fh_Q = gRBM.finalHiddenValues()

brbm = Bernoulli_RBM(fh_Q ,70 ,10 ,randomNumberGenerator ,0.3)
brbm.updations()

conceptSpace_Query = brbm.finalHiddenValues() #CONCEPT SPACE


feh = gRBM.getFE()

plt.plot([1,2],feh,'ro')

plt.xlabel("GIBBS STEP")
plt.ylabel("ENERGY")
plt.show()
print "feh"
print feh

cos_Sims = list()
cosSims_Dict = dict()

for it in xrange(1,listOfArrays[0].__len__()):
    gRBM = Gaussian_RBM(listOfArrays[0][it], 50, 70, randomNumberGenerator, 0.3)
    gRBM.updations()
    fh = gRBM.finalHiddenValues()

    cos_Sim = 1 - spatial.distance.cosine(fh_Q,fh)

    cos_Sims.append(cos_Sim)

    cosSims_Dict[cos_Sim] = it

    brbm = Bernoulli_RBM(fh, 70, 10, randomNumberGenerator, 0.3)
    brbm.updations()

print "COSINE SIMILARITIES"
print cos_Sims
index = cosSims_Dict[max(cos_Sims)]
print cosSims_Dict[max(cos_Sims)]

print documents[0][1]
print documents[0][index]


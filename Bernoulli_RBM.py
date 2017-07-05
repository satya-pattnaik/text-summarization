import numpy as np
from scipy.stats import logistic

class Bernoulli_RBM(object):

    def __init__(self,inputData,numberOfVisibleUnits,numberOfHiddenUnits,randomNumberGenerator,learningRate):

        self.input = inputData

        self.numberOfVisibleUnits = numberOfVisibleUnits
        self.numberOfHiddenUnits = numberOfHiddenUnits
        self.learningRate = learningRate

        # ------------WEIGHTS FOR CONNECTION BETWEEN VISIBLE AND HIDDEN LAYERS----------
        self.weights = np.asarray(
            randomNumberGenerator.uniform(
                low= - float(6/float(self.numberOfVisibleUnits + self.numberOfHiddenUnits)),
                high= float(6/float(self.numberOfVisibleUnits + self.numberOfHiddenUnits)),
                size=(self.numberOfVisibleUnits,self.numberOfHiddenUnits)
            ))

        # ---------------------------BIAS VALUES FOR VISIBLE AND HIDDEN UNITS-------------------------------------

        self.biasForVisibleUnits = np.asarray(
            randomNumberGenerator.uniform(
                low= - float(6/float(self.numberOfVisibleUnits + self.numberOfHiddenUnits)),
                high= float(6/float(self.numberOfVisibleUnits + self.numberOfHiddenUnits)),
                size=(self.numberOfVisibleUnits,)
            )
        )

        self.biasForHiddenUnits = np.asarray(
            randomNumberGenerator.uniform(
                low= - float(6/float(self.numberOfVisibleUnits + self.numberOfHiddenUnits)),
                high= float(6/float(self.numberOfVisibleUnits + self.numberOfHiddenUnits)),
                size=(self.numberOfHiddenUnits,)
            )
        )

    def sampleHiddenValues(self,ip):
        hiddenMean = logistic.cdf(np.dot(np.transpose(self.weights) , ip) + self.biasForHiddenUnits)



        hiddenUnits = list()

        for seed,eachMean in enumerate(hiddenMean):


            randomVariable = np.random.RandomState( seed * 1000 + seed * 20 + seed * 33)
            uniformRandomVariable = randomVariable.uniform()

            if uniformRandomVariable>eachMean:
                hiddenUnits.append(0)
            else:
                hiddenUnits.append(1)

        self.hiddenUnits = np.array(hiddenUnits)

        print "BRBM"
        print self.hiddenUnits.__len__()


    def sampleVisibleValues(self,ip):

        visibleMean = logistic.cdf(np.dot( ip , np.transpose(self.weights) ) + self.biasForVisibleUnits)

        visibleUnits = list()

        for seed,eachMean in enumerate(visibleMean):

            randomVariable = np.random.RandomState( seed * 1000 + seed * 20 + seed * 33)
            uniformRandomVariable = randomVariable.uniform()

            if uniformRandomVariable>eachMean:
                visibleUnits.append(0)
            else:
                visibleUnits.append(1)

        self.visibleUnits = np.array(visibleUnits)



    def updations(self):

        self.freeEnergyList_BRBM = list()

        self.sampleHiddenValues(self.input)
        fe = self.calculateFreeEnergy(self.input)
        print "FREE ENERGY"
        self.freeEnergyList_BRBM.append(fe)
        print fe

        positiveHiddenSamples = self.hiddenUnits

        a = np.reshape(self.input,(1,70))
        b = np.reshape(self.hiddenUnits,(1,10))

        positiveHiddenMeans_Dot_inputVector = np.dot(np.transpose(a) , (b) )

        self.sampleVisibleValues(self.hiddenUnits)
        self.sampleHiddenValues(self.visibleUnits)

        fe = self.calculateFreeEnergy(self.visibleUnits)
        print "FREE ENERGY"
        self.freeEnergyList_BRBM.append(fe)
        print fe

        c = np.reshape(self.visibleUnits,(1,70))
        d = np.reshape(self.hiddenUnits,(1,10))

        negativeHiddenMeans_Dot_negativeVisibleMeans = np.dot(np.transpose(c) , d)

        weightUpdate = (self.learningRate * (
        positiveHiddenMeans_Dot_inputVector - negativeHiddenMeans_Dot_negativeVisibleMeans)) / float(
            self.numberOfVisibleUnits)



        biasOfHiddenUnitsUpdate = (self.learningRate * (positiveHiddenSamples - self.hiddenUnits)) / float(
            self.numberOfHiddenUnits)

        self.biasForHiddenUnits -= biasOfHiddenUnitsUpdate

        biasOfVisibleUnitsUpdate = (self.learningRate * (self.input - self.visibleUnits)) / float(
            self.numberOfVisibleUnits)

        self.biasForVisibleUnits -= biasOfVisibleUnitsUpdate

        print "UPDATION AT BRBM DONE"
        self.weights -= weightUpdate


    def calculateFreeEnergy(self,ip):
        visibleUnits_States = ip
        biasVisible = self.biasForVisibleUnits

        hiddenUnits_States = self.hiddenUnits
        biasHidden = self.biasForHiddenUnits

        weights = self.weights

        hidden_Dot_Weights = np.dot(hiddenUnits_States,np.transpose(weights))

        energy = -np.dot(biasVisible,np.transpose(visibleUnits_States)) - np.dot(biasHidden,np.transpose(hiddenUnits_States)) - np.dot(hidden_Dot_Weights,np.transpose(visibleUnits_States))

        return energy

    def finalHiddenValues(self):
        return self.hiddenUnits
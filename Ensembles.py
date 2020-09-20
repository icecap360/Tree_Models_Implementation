import numpy as np
import Tree
from scipy import stats as s
from statistics import mode, mode
import random

#This module implements bagging models and random forests models
#ntree controls the number of samples and featureesToConsider control number
#of features to consider at a time (needed for random forests)

  
class Ensemble:
    def __init__(self, inputs, labels, featuresToConsider, impurityCat,
                 impurityNum, featureNames, featureTypes, nmin, ntree,
                 maxAlpha, numIntervals=999):
        self.ntree = ntree
        self.labeltype = 'string' if (np.issubdtype(labels.dtype , np.string_) or
            np.issubdtype(labels.dtype , np.object_)) else 'numeric'
        self.maxAlpha = maxAlpha
        self.trees = []
        for i in range(ntree):
            indexes = random.choices(list(range(0,inputs.shape[0])),
                                     k = inputs.shape[0])
            newTree=Tree.Tree(inputs[indexes,:], labels[indexes],
                    featuresToConsider,impurityCat, impurityNum,
                featureNames,featureTypes, nmin, numIntervals)
            newTree.make()
            newTree.prune(maxAlpha)
            self.trees.append(newTree)

    def predict(self,inputs):
        predictions = []
        for t in self.trees:
            predictions.append(t.predict(inputs))
        predictions = np.transpose(np.array(predictions))
        if self.labeltype == 'string':
            def mostCommon(row):
                return s.mode(row)[0][0]#mode(row)
            return np.apply_along_axis(mostCommon,1,predictions)
        else:
            return np.apply_along_axis(mean,1,predictions)
            
            

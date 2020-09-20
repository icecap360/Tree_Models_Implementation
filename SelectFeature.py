import numpy as np
import random
import Decision

def impurityResubLabel(labels , impurityCat, impurityNum, prob=1):
    if (np.issubdtype(labels.dtype, np.string_) or
        np.issubdtype(labels.dtype,np.object_)):
        return impurityCat(labels)*prob
    return impurityNum(labels) * np.var(labels)

def selectBestFeature(inputs, labels, impurityCat,impurityNum,
                      featuresToConsider, featureNames, featureTypes,
                      numIntervals,seed=0):
    random.seed(seed)
    if featuresToConsider>inputs.shape[1] or featuresToConsider<0:
        Exception('featuresToConsider must be between 0 and {maxx}'.format(
            maxx = featuresToConsider))
    possibleFeatures = np.random.choice(featureNames, featuresToConsider, False)
    bestImpurity = np.inf
    bestSplit = ('','','') #in the form (featureName, isCategorical, arguements)

    for name in possibleFeatures:
        colIndex = np.where(name == featureNames)[0][0]
        vals = inputs[:,colIndex]
        isCategorical = False
        if featureTypes[colIndex] == 'string':
                isCategorical=True
        if isCategorical:
            #if the feature is categorical, compute a split for each possible
            #value and pick the maximum one
            subset = [] 
            splitImpurity = bestImpurity
            for cat in np.unique(vals):
                dec = Decision.Categorical(name, [cat])
                trueIndices    = np.vectorize(dec.function)(vals)
                falseIndices   = np.logical_not(trueIndices)
                trueNodeImpurity  = impurityResubLabel(labels[trueIndices],
                        impurityCat, impurityNum,sum(trueIndices)/len(labels))
                falseNodeImpurity = impurityResubLabel(labels[falseIndices],
                        impurityCat, impurityNum,sum(falseIndices)/len(labels))
                if splitImpurity>trueNodeImpurity+falseNodeImpurity:
                    subset = [cat]
                    splitImpurity = trueNodeImpurity+falseNodeImpurity
            split = (name, True, subset )
            
        else:
            #else the feature is numerical. In that case I discretize the
            #continuous features into "numIntervals" bins  or
            #however many intervals are possible.
            
            def findReImpurityGivenBoundary(bound):
                lessBound = labels[vals<=bound]
                moreBound = labels[vals>bound]
                trueNodeReImpurity = impurityResubLabel(lessBound,
                        impurityCat,impurityNum, len(lessBound)/len(labels))
                falseNodeReImpurity =impurityResubLabel(moreBound,
                        impurityCat,impurityNum, len(moreBound)/len(labels))
                return trueNodeReImpurity + falseNodeReImpurity
            numIntervalsForContinuousFeat = min(numIntervals,
                                                    inputs.shape[0])
            possibleBounds = [max(i) for i in np.array_split(np.sort(vals), numIntervalsForContinuousFeat) ]
            possibleBounds = np.array(possibleBounds)
            splitImpurity =  np.inf
            bestBoundary = 0
            for bound in possibleBounds:
                imp = findReImpurityGivenBoundary(bound)
                if splitImpurity > imp:
                    splitImpurity = imp
                    bestBoundary = bound
            split=  (name, False, ('less',bestBoundary) )
        if bestImpurity>splitImpurity:
            bestImpurity = splitImpurity
            bestSplit = split
    return bestSplit
    

#extra code, i originally wanted to a binary search for the the best boundary,
#but instead I have searched discretized the numerical features
#def findKthLargest(vals, index):
#    return np.partition(vals, index)[index-1]
#
#epsilon = 0.000001
#l,r = 0,len(labels)-1
#while r-l>1:
#    median = (l+r)//2
#    bound_median = findKthLargest(vals, median)
#    bound_median_plus_one = findKthLargest(vals, median+1)
#    increaseReImpurity = findReImpurityGivenBoundary(
#        bound_median)-   findReImpurityGivenBoundary(
#            bound_median_plus_one)>0
#    if increaseReImpurity:
#        r = median
#    else:
#        l = median+1
#        
#possibleBounds = np.array([ findKthLargest(vals,l),
#                            findKthLargest(vals,r)])
#possibleBounds = [possibleBounds[0]-epsilon,
#                  possibleBounds[1]+epsilon,
#                  possibleBounds[1],possibleBounds[1]]
#bestBoundary = possibleBounds[np.argmin(np.vectorize(
#    findReImpurityGivenBoundary)(possibleBounds))]

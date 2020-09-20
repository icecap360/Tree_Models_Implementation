import numpy as np
import Node
import SelectFeature
def build(node, entireDatasetLength, impurityCat, impurityNum,
          featureNames, featureTypes, featuresToConsider, nmin ,
          numIntervals, forceToTerminal=False):
    #selects best feature, and then computes resubstitution impurity of the node
    #the complete impurity can only be calculated later when we know the
    #subtree size of each node (for that the entire tree must be built)
    if node.nodeClass != 'unexplored':
        raise Exception('The node {des} is already built/explored'.format(
            des = node.__repr__() ))

    #find the probability of being in the node and the resubstitution error
    #resubstirution error is calculated differently for categorical vs
    #numerical labels
    probabilityOfBeingInNode = len(node.labels)/entireDatasetLength
    if node.getLabelType() == 'string':
        impurity = impurityCat(node.labels)
        resubstitutionImpurity = impurity*probabilityOfBeingInNode
    else:
        impurity = impurityNum(node.labels)
        resubstitutionImpurity = impurity*np.var(node.labels)

    #check if the number of examples is less then the predefined minimum
    #or if the node already pure (I round to zero impurity if the impurity
    # is less then 10^-10)
    if (forceToTerminal or node.inputs.shape[0]<=nmin or
        resubstitutionImpurity <= 10**-10):
        if (node.getLabelType() == 'string'):
            terminalNode = Node.ClassificationTerminal(node.labels)
        else:
            terminalNode = Node.RegressionTerminal(node.labels)
        terminalNode.probability = probabilityOfBeingInNode
        terminalNode.impurity = resubstitutionImpurity
        return terminalNode
    
    #otherwise we need to make a nonterminal node
    featureName, isCategorical,arguments = SelectFeature.selectBestFeature(
        node.inputs, node.labels, impurityCat,impurityNum, featuresToConsider,
        featureNames, featureTypes, numIntervals)
    if isCategorical:
        nonTerminalNode = Node.NonTerminal(node.inputs,node.labels,
                                      featureName, featureNames, featureTypes,
                                           subset = arguments[0])
    else:
        nonTerminalNode = Node.NonTerminal(node.inputs, node.labels, featureName,
                                           featureNames,featureTypes,
                                           operator=arguments[0],
                                           boundary=arguments[1])
    nonTerminalNode.probability = probabilityOfBeingInNode
    nonTerminalNode.impurity = resubstitutionImpurity
    return nonTerminalNode 

import numpy as np
from scipy.stats import entropy
import Node
import BuildNode


class Tree:
    def __init__(self, inputs, labels, featuresToConsider, impurityCat,
                 impurityNum, featureNames, featureTypes, nmin, numIntervals=999):
        self.inputs = inputs
        self.labels = labels
        self.impurityCat = impurityDict[impurityCat]
        self.impurityNum = impurityDict[impurityNum]
        self.featuresToConsider = featuresToConsider
        self.featureNames = featureNames
        self.featureTypes = featureTypes
        self.nmin = nmin
        self.root = Node.Node(inputs, labels, '')
        self.numIntervals = numIntervals
        self.lenOfData = self.inputs.shape[1]
    def make(self):
        #I will build the tree in using BFS, each element in the queue
        #will be a nonterminal node that is built but whose children are not
        #built
        self.root = BuildNode.build(self.root, self.lenOfData,self.impurityCat,
                                    self.impurityNum, self.featureNames,
                                    self.featureTypes, self.featuresToConsider,
                                    self.nmin, self.numIntervals)
        if isinstance(self.root, Node.Terminal):
            return
        queue = [self.root]
        while (len(queue)>0):
            nodee = queue[0]
            queue.pop(0)
            nodee.trueNode = BuildNode.build(nodee.trueNode, self.lenOfData,
                                             self.impurityCat,
                                    self.impurityNum, self.featureNames,
                                    self.featureTypes, self.featuresToConsider,
                                self.nmin, self.numIntervals)
            nodee.falseNode = BuildNode.build(nodee.falseNode, self.lenOfData,
                                             self.impurityCat,
                                    self.impurityNum, self.featureNames,
                                    self.featureTypes, self.featuresToConsider,
                                self.nmin, self.numIntervals)
            if (isinstance(nodee.trueNode , Node.NonTerminal)):
                queue.append(nodee.trueNode)
            if (isinstance(nodee.falseNode , Node.NonTerminal)):
                queue.append(nodee.falseNode)

    def draw(self):
        #This function visualizes the tree by printing it to the console
        #I will print the tree using breadth first search, the queue will
        #contain a tuple of the node and the level of the node
        with open('TreeDrawing.txt', 'w') as f:
            queue = [(self.root,0)]
            currentLevel=-1
            while len(queue) != 0:
                nextt = queue.pop(0)
                nodee = nextt[0]
                levell=nextt[1]
                if (levell !=currentLevel):
                    f.write('NEW LEVEL \n')
                    currentLevel=levell
                f.write('Level: '+ str(levell)+ ' NODE: '+ nodee.__repr__()+'\n')
                if isinstance(nodee, Node.NonTerminal):
                    queue.append((nodee.trueNode,levell+1 ))
                    queue.append((nodee.falseNode, levell+1))

                
    def predict(self,data):
        predictions = []
        for i in range(data.shape[0]):
            pred = None
            currentNode = self.root
            while isinstance(currentNode, Node.NonTerminal):
                colIndex = np.where(currentNode.decision.featureName ==
                                    self.featureNames)[0][0]
                if currentNode.decision.function(data[i,colIndex]): 
                    currentNode = currentNode.trueNode
                else:
                    currentNode = currentNode.falseNode
            predictions.append(currentNode.prediction)
        return predictions

    
    def prune(self, maxAlpha):
        #repeatedly find the weakest node (the node with the smallest alpha)
        #and remove that weakest node if its alpha is less then maxalpha
        #hyperparameter
        smallestAlpha = np.inf
        smallestAlphaNode = None #will store node to be pruned
        smallestAlphaNodeParent = None #will store the parent of the node to be pruned
        def recursePrune(currentNode, parentNode):
            #returns the weakest node
            nonlocal smallestAlpha
            nonlocal smallestAlphaNodeParent
            nonlocal smallestAlphaNode
            if isinstance(currentNode, Node.NonTerminal):
                #computes number of leaves and tree impurity of currentNode
                leavesTr, sumImpurityTr=recursePrune(currentNode.trueNode, currentNode)
                leavesFls, sumImpurityFls=recursePrune(currentNode.falseNode, currentNode)
                leaves = leavesTr + leavesFls
                sumImpurity = sumImpurityTr + sumImpurityFls

                #update node to be pruned if this node is worse
                alpha = (currentNode.impurity-sumImpurity)/(leaves-1)
                if alpha < smallestAlpha:
                    smallestAlpha = alpha
                    smallestAlphaNodeParent = parentNode
                    smallestAlphaNode = currentNode 
                return leaves, sumImpurity
            else:
                #if the node is terminal node, then it has one leaf
                return 1 , currentNode.impurity
        recursePrune(self.root, None)
        while (smallestAlpha < maxAlpha):

            if smallestAlphaNode is None or smallestAlphaNode is self.root:
                raise Exception('Cannot prune the root, maxalpha is too high')
            
            #many of the features are not applicable, I have replaced them with Nones
            newNode = BuildNode.build(Node.Node(None,smallestAlphaNode.labels,''),
                    self.lenOfData,self.impurityCat, self.impurityNum,
                    None, None,None, None,None, forceToTerminal=True)
            if smallestAlphaNode is smallestAlphaNodeParent.trueNode:
                smallestAlphaNodeParent.trueNode = newNode
            else:
                smallestAlphaNodeParent.falseNode = newNode
            smallestAlpha = np.inf
            smallestAlphaNodeParent = None
            recursePrune(self.root, None)
            
            
        
impurityDict = {
        'entropy':(lambda labs: zeroCase( getProbs(labs),entropy)),
        'gini':(lambda labs: zeroCase(getProbs(labs), Gini)),
        'mse':(lambda labs: zeroCase(labs, mse))
        }
def zeroCase(labs, f):
    if len(labs):
        return f(labs)
    else:
        return 0 
def Gini(probs):
    return (1-np.sum(probs**2))
def mse (labs):
    return np.mean( (labs - np.mean(labs))**2 )
            
def getProbs(arr):
    _,counts = np.unique(arr, return_counts=True)
    return counts/np.sum(counts)
    

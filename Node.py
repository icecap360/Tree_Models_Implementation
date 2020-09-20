import numpy as np
import Decision

class Node:
    def __init__(self, inputs, labels,description,nodeClass='unexplored' ):
        #in the initial state the node is always unexplored and only contains
        #raw data
        self.inputs = inputs
        
        if type(labels) is list:
            labels = np.array(labels)
        self.labels = labels
        
        self.nodeClass = nodeClass
        self.description = description

        #Node probability and Impurity are calculated after the node is built
        self.probability = 0
        self.impurity=0
    def __repr__(self):
        return self.description
    def getLabelType(self):
        if (np.issubdtype(self.labels.dtype , np.string_) or
            np.issubdtype(self.labels.dtype , np.object_)):
            return 'string'
        return 'numeric'


    
class Terminal(Node):
    #Terminal nodes have a prediction attatched to them along with a
    #I could calculate the impurity of a terminal node right away, but it would be
    #clearer if I calculated all impurities (including nonterminal nodes)
    #at the end
    def __init__(self, labels, prediction):
        self.prediction = prediction
        des = "Terminal: {pred}".format(pred = self.prediction)
        Node.__init__(self, [], [], des, 'terminal')
class ClassificationTerminal(Terminal):
    def __init__(self, labels):
        #Predict the most common class (predict none if the node is empty)
        #We also need to store the raw class probabilities
        if len(labels) == 0:
            Terminal.__init__(self, labels, None)
            return
        unique, positions = np.unique(labels, return_inverse=True) 
        counts = np.bincount(positions)
        maxpos = counts.argmax()            
        prediction = unique[maxpos]
        self.classProbabilities = counts / np.sum(counts)
        Terminal.__init__(self, labels, prediction)
class RegressionTerminal(Terminal):
    def __init__(self, labels):
        #Predict the mean of all labels in the node
        if len(labels) == 0:
            Terminal.__init__(self, labels, None)
            return
        Terminal.__init__( self, labels, np.mean(labels) )



class NonTerminal(Node):
    def __init__(self, inputs, labels, featureName,featureNames,featureTypes,
                 subset=[], boundary=None,operator = ''):
        #Find the actual feature array based on the feature name
        colIndex = np.where(featureName == featureNames)[0][0]
        features = inputs[:,colIndex]
        #The key feature of noonterminal nodes is that they have a decision
        if featureTypes[colIndex] == 'string':
            #then the feature is of type categorical and the decision must be categorical
            self.decision = Decision.Categorical(featureName, subset)
        else:
            #else the feature is of type numerical and decision must be numerical
            self.decision = Decision.Numerical(featureName, operator, boundary)

        #create the true and false nodes, both unexplored, by filtering the
        #data to each node based on the decision 
        trueIndices    = np.vectorize(self.decision.function)(features)
        falseIndices   = np.logical_not(trueIndices)
        self.trueNode  = Node(inputs[trueIndices],labels[trueIndices],'')
        self.falseNode = Node(inputs[falseIndices],labels[falseIndices],'')

        #the node will never use its input field again, but the labels field
        #is needed for tree pruning
        Node.__init__( self, [], labels,
                           self.decision.__str__(),'decision' )

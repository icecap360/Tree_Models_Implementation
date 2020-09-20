import numpy as np
class Decision:
    def __init__(self, function, printString, featureName):
        self.function = function
        self.printstring = printString
        self.featureName=featureName
    def __str__(self):
        return self.printstring

class Categorical(Decision):
    def __init__(self, featureName, subset):
        self.subset = subset
        
        function = lambda x : np.isin(x, np.array(subset))

        printString = "({featureName} is an element of {listOfElem})".format(
            featureName=featureName, listOfElem=str(self.subset))
        
        Decision.__init__(self, function, printString, featureName)

class Numerical(Decision):
    def __init__(self, featureName, operator, boundary):
        self.operator=operator
        self.boundary=boundary

        if operator == 'less':
            self.function = lambda x: np.less_equal(x,self.boundary)
        else:
            self.function = lambda x: np.greater_equal(x,self.boundary)
                              
        printString = "({featureName} is {oper} then {bound})".format(
            featureName=featureName, oper=operator,bound=str(boundary))
                              
        Decision.__init__(self, self.function, printString, featureName)

        
        
    

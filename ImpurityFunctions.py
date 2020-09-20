import numpy as np
class Decision:
    def __init__(self, function, printString):
        self.function = function
        self.printstring = printString
    def __str__(self):
        return self.printstring

class Categorical(Decision):
    def __init__(self, featureName, subset):
        self.subset = subset
        self.featureName = featureName
        
        function = lambda x : np.isin(x, self.subset)
        
        printstring = "({featureName} is an element of {listOfElem})".format(
            featureName=sfeatureName, listOfElem=str(self.subset))
        
        super().__init__(self, function, printString)

class Numerical(Decision):
    def __init__(self, featureName, operator, boundary):
        self.featureName=featureName
        self.operator=operator
        self.boundary=boundary

        operatorConversions ={'less': (lambda x,y: np.less(x,y)},
                  'greater': (lambda x,y: np.greater(x,y)}
        self.function = lambda x: operatorConversions[operator](x,boundary)
                              
        printstring = "({featureName} is {oper} then {boundary})".format(
            featureName=featureName, oper=operator,boundary=str(boundary))
                              
        super().__init__(self, function, printString)

        
        
    

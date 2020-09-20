import numpy as np
import pandas as pd
import Tree as tt
from sklearn.preprocessing import MinMaxScaler
data = {'HomeOwner': ['Yes', 'No','No','Yes'
                      ,'No','No','Yes',
                      'No','No','No'],
        'Maritial Status': ['Single','Married',
                            'Single','Married',
                            'Divorced','Married',
                            'Divorced','Single',
                            'Married','Single'],
        'Income':[125,100,70,120,95,60,220,85,75,90],
        'Defaulted':['No', 'No', 'No','No',
                     'Yes','No','No','Yes',
                     'No','Yes']
        }
def normalizeCol(col):
    scalar = MinMaxScaler()
    col = np.reshape(col, (-1, 1))
    col = (col-np.mean(col))/np.sqrt(np.var(col))
    return col
def getType(array):
    if np.issubdtype(array.dtype,np.string_):
        return 'string'
    else:
        return 'numeric'
def findAndFixFeatureTypes(data):
    #returns a array of types of each column
    types = []
    for i in data.columns:
        vals = data[i].to_numpy()
        if np.issubdtype(data[i].dtype, np.object_):
            vals = vals.astype('S')
        types.append(getType(vals))
    return types, data
data =  pd.DataFrame(data=data)
data['Income'] = normalizeCol(data['Income'].values.astype(float))
print(data)
labels = data.pop('Defaulted').to_numpy().astype('S')
labelType = getType(labels)

featureNames = data.columns.to_numpy().astype('S')
featureTypes, data = findAndFixFeatureTypes(data)
inputs=data.to_numpy()

tree = tt.Tree(inputs, labels, inputs.shape[1],
               'entropy', 'mse',featureNames,featureTypes, 1)

tree.make()
tree.draw()
print(tree.predict(inputs))




##
##
##    
##    #(HomeOwner?)(Maritial status)(income)(defaulted)
##    ['Yes',         'Single',       125,    'No'],
##    ['No',          'Married',      100,    'No'],
##    ['No',          'Single',       70,     'No'],
##    ['Yes',         'Married',      120 ,   'No'],
##    ['No',          'Divorced',     95,     'Yes'],
##    ['No',          'Married',      60,     'No'],
##    ['Yes',         'Divorced',     220,    'No'],
##    ['No',          'Single',       85,     'Yes'],
##    ['No',          'Married',      75,     'No'],
##    ['No',          'Single',       90,     'Yes'],
##    ])

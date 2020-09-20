import numpy as np
import pandas as pd
import Tree as tt
import Ensembles as Ensembles
from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv('vehicle.csv')
print('SOME SAMPLE DATA')
print(data.head())
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
#data['Income'] = normalizeCol(data['Income'].values.astype(float))
labels = data.pop('V19').to_numpy().astype('S')
labelType = getType(labels)

featureNames = data.columns.to_numpy().astype('S')
featureTypes, data = findAndFixFeatureTypes(data)
inputs=data.to_numpy()

tree = tt.Tree(inputs, labels, inputs.shape[1],
               'entropy', 'mse',featureNames,featureTypes, 1, 10)

tree.make()
print('DRAWING TREE')
tree.draw()
print('TESTING IF THE TREE CAN OVERFIT')
print('A table showing predictions vs labels, they should match each other')
predsLab = pd.DataFrame({
    'predictions' :tree.predict(inputs),
    'actual labels':labels,
    'are they same': tree.predict(inputs) == labels
    })
print(predsLab)
tree.prune(0.5)
tree.draw()
print(np.mean(tree.predict(inputs) == labels))

ens = Ensembles.Ensemble(inputs, labels,10,
               'entropy', 'mse',featureNames,featureTypes, nmin=1, ntree=5,
                         maxAlpha=0.1, numIntervals=10)
print(ens.predict(inputs))

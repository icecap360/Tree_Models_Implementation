# Tree_Models_Implementation

Using object oriented design, this repository implements a tree based learning algorithms and provides support for ensemble learning. 

[Decision tree learning](https://en.wikipedia.org/wiki/Decision_tree_learning) uses widely known tree data structure to construct a machine learning model for supervised tasks 
(supervised tasks are those with predefined labels). For example, using data about a specific car to predict if its type (van, regular, bus, etc.), which is also one of the test cases.
The following is a tree model for predicting animal species:

![alt text](https://cdn-images-1.medium.com/max/824/0*J2l5dvJ2jqRwGDfG.png)

More formally a decision tree contains:
* Internal nodes that contains a boolean question
* A "true" child node if the parent node's boolean question answers true 
* A "false" child node if the parent node's boolean question answers true 
* Terminal node, contains the final label to predict, i.e. a class for classification problems and numerical value for regression problems.

Furthermore ensemble methods, which are machine learning methods that build multiple models using samples from the full dataset, and then making predictions by taking the mean
or mode of what each of the models predicts (sort of like democraticly voting). For specifically I implement 
[bagged decision trees](https://en.wikipedia.org/wiki/Bootstrap_aggregating#:~:text=Bootstrap%20aggregating%2C%20also%20called%20bagging,and%20helps%20to%20avoid%20overfitting.), 
and [random forests](https://en.wikipedia.org/wiki/Random_forest) in python. 

Technology
* Python (this project was "built from scratch")
* Pandas and numpy for data preprocessing

Test cases
* Test_case_1: using data about credit card holder to predict if they will default
* Test_case_2: using data about a specific car to predict if its type (van, regular, bus, etc.)

How the algorithm works:
Before starting the algorithm, one much pick the following hyperparameters
- nmin: the minimum number of training examples an internal node can have before it must become a terminal node
- alpha: after the tree is built, this parameter controls the amount of pruning (removal of subtrees and converting internal nodes to terminal nodes) we do.
- impurity function: a function for for telling how "impure" or "mixed" a node is. For example in regression tasks the impurity is the variance of the labels, or in classification tasks
the impurity could be the entropy of the labels within that node.
- ntree: how many tree models to make
- number of features to consider: This limits the number of possible boolean questions a node can have, only needed is the random forests implementation is chosen. More specifically a node can only pick 
"number of features to consider" columns in the input, and the boolean question can only use information from those columns.

Another thing to note is that every node contains the following information: inputs (drop all the training data down the decision tree, what is the features of instances that use this node
in there path), labels (drop all the training data down the decision tree, what is the label of every instance of data that uses this node in there path), impurity (calculated from the labels)

1. Initialize a root node with class "unexplored" and all the data.
2. Pick any "unexplored" node. 
3. "Build" the node, which is basically calculate the node impurity and then 
- If the number of instances of data is less then nmin, convert the "unexplored" node to a "terminal" node with prediction as the most mean or mode of the labels of that node (mode if classification mean if regression).
- Else convert the node to a "nonterminal" node, by making a greedy decision: pick the boolean question that would minimize the impurity of the two children. Make 2 new "unexplored nodes", 
a true child node and false child node. Apply the boolean function to every input label pair, and send them to the true or false node accordingly. Note that the boolean function must be picked
subject to the constraints of the "number of features to consider".
4. While there is an unexplored node, do step 3 on it.
5. Prune the tree: the algorithm is complicated, but in a nut shell if the decrease in impurity of a subtree (impurity of root - sum of impurity of terminal nodes) is not large enough, remove the subtree
6. Take a bootstrap sample (for example if there were N instances in the whole dataset, a bootstrap would be to take N samples with replacement). Build an tree (steps 1-5) on that sample.
Do this ntree times.
7. Make predictions on a test instance by averaging (for regression tasks) or taking the mode(for classification tasks) the predictions of the ntree models.

Design
File structure

Class diagram

# Here are some tricks to keep in mind.

**Segregate Train and Test data**
```
from sklearn import cross_validation
### test_size is the percentage of events assigned to the test set
### (remainder go into training)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)
```

**Pickle it**
```
import pickle
mydate = pickle.load(open('filename.pkl','rb'))
# cpickle is faster: python3 loads cpickle automatically when you load pickle
```

**Convert raw text to vector : tf-idf**

tf-idf : term frequency - inverse document frequency
```
from sklearn.feature_extraction.text import TfidfVectorizer
### text vectorization--go from strings to lists of numbers
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
features_train_transformed = vectorizer.fit_transform(features_train)
features_test_transformed  = vectorizer.transform(features_test)
```
**Feature selection**
Since text data is highly demensional you should choose only relevant features out of it.
sklearn's ```SelectPercentile``` class automatically does feature selection based on the relevance of
that word with the target

If you decide to select a variable by its level of association with its target, 
the class SelectPercentile provides an automatic procedure for keeping only a certain percentage of the best,
associated features. The available metrics for association are

**f_regression**: Used only for numeric targets and based on linear regression performance.

**f_classif**: Used only for categorical targets (classification applications) and based on the Analysis of Variance (ANOVA) statistical test.

**chi2**: Performs the chi-square statistic for categorical targets, which is less sensible to the nonlinear relationship between the predictive variable and its target.

*Note the features used below for selections are first vectorized.*
```
from sklearn.feature_selection import SelectPercentile, f_classif
### feature selection, because text is super high dimensional and 
### can be really computationally chewy as a result
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(features_train_transformed, labels_train)
features_train_transformed = selector.transform(features_train_transformed).toarray()
features_test_transformed  = selector.transform(features_test_transformed).toarray()
```
**Finding accuracy of a classfication model**
```
import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
accuracy_score(y_true, y_pred)
0.5
accuracy_score(y_true, y_pred, normalize=False)
2
```
# A simple Naive Bayes Classifier
```
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf = clf.fit(features_train, labels_train)
pred_labels = clf.predict(features_test)

from sklearn.metrics import accuracy_score
print accuracy_score(labels_test, pred_labels)
```
**Time your code**
```
from time import time
t0_fit = time()
clf = clf.fit(features_train, labels_train)
print "time to fit the model:",round(time()-t0_fit,3),"s"
```

# SVM : Support Vector Machine
They are all about maximizing the distance of the line from the nearest data point.
That distance is called margin. SVMs maximize the margins.

*A basic SVM implementation*
```
from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy_score(pred, labels_test) #remember to import accuracy_score from sklearn.metrics
```
**kernels** can be any of the following :-
1. linear
2. polynomial
3. rbf : Radial Basis Function
4. sigmoid
5. custom / callable

Advantages :-
1. effective in high demensional spaces
2. efective even when the dimensions are greater than the number of samples
3. uses a subset of training points so it's memory efficient
4. custom kernels can be used

Disadvantages :-
1. poor performance when the number of features is less than the number of samples
2. do not provide probabilty estimates

SVMs decision function depends upon the support vectors which are a subset of the training sample
```
# get support vectors
clf.support_vectors_
# get indices of support vector
cls.support_
# get number of support vectors for each class
clf.n_support_
```
**SVM Parameters**
1. C parameter - Panalty parameter of the model. The larger the value of C means you'll get more training set correctly classfied and that the decision boundary will be tightly coupled with the training set. Therefore there is a trade off for the right amount of panalty you wan't to implement.
2. Gamma parameter - How far the influence of a single training parameter reach. 
A high value of gamma means that only the points close to the decision boundary take part in deciding where the decision boundary should be. In this case the decision boundary seems to be more tight fitted.
A Low value of gamma means that even the points farther away plays infuence where the decision boundary is going to be. in this case the decision boundary seems to be more linear.

The kernel, C parameter and Gamma parameter together play the role in over-fitting the model.

**When to use Naive Bayes and when to go with SVM**

SVMs work great with complicated situations where there is a clear margin between the classes.

They don't work well with large data sets with per say cubic or higher level of separation.
They also don't work well where there is a lot noise.
In these cases Naive Bayse will be more suitable.

For eg : in case of handling text Nave Bayes will fare better.

Accuracy level with different parameters with respect to the udacity email classification dataset:-
```
kernel='linear' : highest appx 0.98
kernel='rbf' :    low appx 0.68
C = 10.0     :      0.61
C= 100.0     :      0.61
C= 1000.0     :     0.821
C= 10000.0    :     0.892
```
So basically the accuracy increased with the Value of C (the panalty parameter) - fitting closely with the training set

# Decision Trees
Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

Used for non-linear classifications & regressions which requires different decisions paths to be taken depending on the previous choice.
```
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
```
*Parameters of Decision Tree Classifiers*

**min_samples_split** : the min number of samples that should be present in the node to allow further splitting. Used to prevent overfitting of the data.
                    Therefore, for eg, with a min_samples_split = 2 you can't split a node with only 1 sample left. The                         default values is 2. Generally low min_samples_split values lead to over-fitting. So high values like 50 or so would be better than 2.
                    
**Entropy** : comtrols how a DT decides where to split the data. It's a measure of impurity in a bunch of data samples.
Basically while making a DT you are trying to find variables that devide the data into subsets which are as pure as possible.

Entropy - summation of ( -pi . log base2 (pi)) where pi are the data points belonging to a differnt class in that subset.
For eg, if all the data points in a subset belong to the same class then the purity is max and entropy is 0.
On the other hand if we have 2 class labels and the data points are evenly split between the two classes then the entropy is max = 1

**Calculation of Entropy example**

lets say a node has 4 data points |SSFF| 
```
total points T = 4
S points = 2
F points = 2
so, ps = 2/4 = 0.5
&   pf = 2/4 = 0.5

and entropy is 
-ps.log2(ps) - pf.log(pf)

-0.5*-1 -0.5*-1 = 1
```
So we get a data set where there are 2 classes and the data points are evenly split between the two so the entropy is max.

**Information Gain**

IG = entropy(parent) - [ weighted avg ] .entropy (children)

DT will try to maximize the information gain.
Clearly if the children nodes have aentropy 1 and the information gain is going to be 0. That parent node is clearly **not** the place where you want to start splitting your decision boundary. 
Remember : DT continously try to maximize the information gain while splitting for a boundary.

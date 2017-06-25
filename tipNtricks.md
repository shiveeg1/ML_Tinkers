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
**A simple Naive Bayes Classifier**
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
t0_fit = time()
clf = clf.fit(features_train, labels_train)
print "time to fit the model:",round(time()-t0_fit,3),"s"
```

**SVM : Support Vector Machine**
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
3. rbf
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


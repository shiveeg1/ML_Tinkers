# Here are some tricks to keep in mind.

*Segregate Train and Test data*
```
from sklearn import cross_validation
### test_size is the percentage of events assigned to the test set
### (remainder go into training)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)
```

*Pickle it*
```
import pickle
mydate = pickle.load(open('filename.pkl','rb'))
# cpickle is faster: python3 loads cpickle automatically when you load pickle
```

*Convert raw text to vector : tf-idf *
tf-idf : term frequency - inverse document frequency
```
from sklearn.feature_extraction.text import TfidfVectorizer
### text vectorization--go from strings to lists of numbers
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
features_train_transformed = vectorizer.fit_transform(features_train)
features_test_transformed  = vectorizer.transform(features_test)
```
* Feature selection*
Since text data is highly demensional you should choose only relevant features out of it.
sklearn's SelectPercentile class automatically does feature selection based on the relevance of
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


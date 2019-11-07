## learning Data Analytics using the iris flowers dataset.

# load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

# load dataset from url
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# Get a quick idea of how many instances (rows) & attributs (columns) the data
# contains with shape property
print(dataset.shape)

# To peek at the starting from top (head)
print(dataset.head(20))

# peek at the last 20 lines from bottom (tail)
print(dataset.tail(20))

# Look at summary of each attribute (columns)
print(dataset.describe())

# look at the number of instances (rows) class distribution
print(dataset.groupby('class').size())

# promp user to proceed to Data Visualization
print('would you like to continue to data visualization, yes or no')
answer = input()
if answer == 'yes':

    # box and whisker plot. A type of Univariate plot to better understand each
    # attribute
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()

    # create a histogram of each input variable to get an idea of the distribution
    dataset.hist()
    plt.show()

    # look at scatterplots of all pairs of attributes to spot structured relationships between input variables.
    scatter_matrix(dataset)
    plt.show()

else:
    print(':(')

print("evaulate the data with algorithms")
answer2 = input()
### EVAULATE ALGORITHMS IN BATCHES

##CREATE A VALIDATION DATASET
# Create a validation dataset by splitting it into 2 parts. One part 80% to train model. Second part 20% to validate.
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


## TEST HARDNESS
# Test options and evaluation metrics using a 10-fold cross validation. It splits dataset into 9 parts to train and 1 part to test.
seed = 7
scoring = 'accuracy'

##BUILD MODELS
# identify and evualate which algorithm would work best. As well as their configurations.
# Spot Check Algorithms
models = []
models.append(('LB', LogisticRegression(solver='liblienear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# run each model
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# compare algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


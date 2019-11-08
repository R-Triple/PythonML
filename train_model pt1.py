## Unsupervised learning method to 
from sklearn.externals import joblib

# load the data sets
df = pd.read_csv=("ml_house_data_set.csv")

# remove the fileds from the data set that we don't want to include in our model.
del df['house_number']
del df['unit_number']
del df['street_name']
del df['zip_code']

# replace categorical data with one-hot encoded data
features_df = pd.get_dummies(df, columns=['garage_type', 'city'])

#remove the sale price from the feature data
del features_dfl['sale_price']

#create the X and y arrays using .as_matrix to make the data set compatible with numpy
X = features_df.as_matrix()
y = features_df['sale_price'].as_matrix()

# Split the data set in a training set 70% to 30% for our test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# fit regression model with reasonable guesses. 
model = ensemble.GradientBoostingRegressor(
    n_estimators = 1000, # tells the model how much trees to build. Higher takes longer is more accurate. This is a guesss.
    learning_rate = 0.1,
    max_depth = 6,
    min_samples_leaf = 9,
    max_features = 0.1,
    loss = 'huber',
    random_state = 0,
)
model.fit(X_train, y_train)

# Save the trained model to a file for future use
joblib.dump(model, 'trained_house_model.pkl')

# find the error rate on the training sets
mse = mean_absolute_error(y_train, model.predict(X_train))
print("Traning Set Mean Absolute Error: %.4f" % mse)

# find the error rate on the test sets
mse = mean_absolute_error(y_test, model.predict(X_test))
print("Traning Set Mean Absolute Error: %.4f" % mse)

use_model = joblib.load('trained_house_model.pk1)


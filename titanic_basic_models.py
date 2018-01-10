# Data names: train_x, train_y, valid_x, valid_y

m_train, _ = train_y.shape
m_valid, _ = valid_y.shape


# Linear Regression:
from sklearn.linear_model import LinearRegression
lin_class = LinearRegression()
lin_class.fit(train_x,   train_y)

train_lin_pred = (lin_class.predict(train_x) > 0.5).astype(int)
train_lin_acc = np.sum(np.equal(train_lin_pred, train_y)) / m_train

valid_lin_pred = (lin_class.predict(valid_x) > 0.5).astype(int)
valid_lin_acc = np.sum(np.equal(valid_lin_pred, valid_y)) / m_valid


# Decision Tree:
from sklearn.tree import DecisionTreeClassifier
tree_class = DecisionTreeClassifier()
tree_class.fit(train_x,  train_y)

train_tree_pred = (tree_class.predict(train_x) > 0.5).astype(int).reshape([-1, 1])
train_tree_acc = np.sum(np.equal(train_tree_pred, train_y)) / m_train

valid_tree_pred = (tree_class.predict(valid_x) > 0.5).astype(int).reshape([-1, 1])
valid_tree_acc = np.sum(np.equal(valid_tree_pred, valid_y)) / m_valid


# Cross-validation
def display_scores(scores):
    print("Scores:",    scores)
    print("Mean:",  scores.mean())
    print("Standard deviation:",    scores.std())

from sklearn.model_selection import cross_val_score
tree_scores = cross_val_score(tree_class,   train_x,   train_y, 
                        scoring="accuracy",   cv=10)

display_scores(tree_scores)


# Random Forest
from sklearn.ensemble import RandomForestClassifier
forest_class = RandomForestClassifier()
forest_class.fit(train_x, train_y)

forest_scores = cross_val_score(forest_class, train_x, train_y, 
                        scoring="accuracy", cv=10)

display_scores(forest_scores)


# # Saving models for later:
# from sklearn.externals    joblib
# joblib.dump(my_model,   "my_model.pkl")
# #   and later...
# my_model_loaded =   joblib.load("my_model.pkl")


# FINE TUNING
# Grid search:

train_x = full_train_x
train_y = full_train_y

forest_class = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV
param_grid = [
    {'bootstrap': [True],  'n_estimators': [3, 7, 10, 15, 30], 'max_features': [2, 4, 5,  6, 7,  8]},
    {'bootstrap': [False], 'n_estimators': [3, 10],'max_features': [2, 3,  4]}]

grid_search = GridSearchCV(forest_class, param_grid, cv=5,
                           scoring='accuracy')

# Expects a 1D array for y (shape = [m,]) and NOT a column vector (shape = [m,1])
grid_search.fit(train_x, np.ravel(train_y))
grid_search.best_params_
grid_search.best_estimator_

# We can look over all results:
cvres = grid_search.cv_results_
for mean_score, params  in  zip(cvres["mean_test_score"],   cvres["params"]):
    print(mean_score, params)

# Now that we have our best estimator:
forest_class_cv = grid_search.best_estimator_
forest_class_cv.fit(train_x, train_y)

forest_scores_cv = cross_val_score(forest_class_cv, train_x, train_y, 
                        scoring="accuracy", cv=10)
display_scores(forest_scores_cv)

forest_pred = forest_class_cv.predict(test)
#forest_acc = np.sum(np.equal(train_tree_pred, train_y)) / m_train


submission = pd.DataFrame({
        "PassengerId": test_passenger_id,
        "Survived": forest_pred
    })
# submission.to_csv('./output/rndforest_submission.csv', index=False)
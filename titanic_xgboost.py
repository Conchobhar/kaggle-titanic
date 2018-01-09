from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# fit model no training data
model = XGBClassifier()
model.fit(train_x, np.ravel(train_y))

# make predictions for test data
train_pred = [round(value) for value in model.predict(train_x)]
valid_pred = [round(value) for value in model.predict(valid_x)]

# evaluate predictions
acc_train = accuracy_score(train_y, train_pred)
acc_test = accuracy_score(valid_y, valid_pred)
print("Train Accuracy: %.2f%%" % (acc_train * 100.0))
print("Test Accuracy: %.2f%%" % (acc_test * 100.0))

# fit model no training data
model_full = XGBClassifier()
model_full.fit(full_train_x, np.ravel(full_train_y))

# make predictions for test data
full_train_pred = [round(value) for value in model_full.predict(full_train_x)]
full_valid_pred = [round(value) for value in model_full.predict(test)]

# evaluate predictions
full_acc_train = accuracy_score(full_train_y, full_train_pred)
print("Train Accuracy: %.2f%%" % (full_acc_train * 100.0))


submission = pd.DataFrame({
        "PassengerId": data_test_passenger_id,
        "Survived": full_valid_pred
    })
# submission.to_csv('/home/xu/Work/kaggle/titanic/output/xgbpy_submission.csv', index=False)

# To try

# ahmeds approach
# matts approach
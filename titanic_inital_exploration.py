################################################################################
# Data Exploration
################################################################################
# See here for more:
# http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
def data_exploration(data_train):
    titanic_dir_train = "/home/xu/Work/kaggle/titanic/data/train.csv" 
    titanic_dir_test  = "/home/xu/Work/kaggle/titanic/data/test.csv"
    data_train = pd.read_csv(titanic_dir_train)
    data_test  = pd.read_csv(titanic_dir_test)

    print(data_train.columns.values)
    print(data_train.head())
    data_train.info()
    print('_'*40)
    data_train.describe()

    # Pivoting data to compare
    data_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    data_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    data_train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    data_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

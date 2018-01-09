from    sklearn.model_selection import  GridSearchCV
param_grid  =   [
                {'n_estimators':    [3, 10, 30],    'max_features': [2, 4,  6,  8]},
                {'bootstrap':   [False],    'n_estimators': [3, 10],    'max_features': [2, 3,  4]},
        ]
forest_reg  =   RandomForestRegressor()
grid_search =   GridSearchCV(forest_reg,    param_grid, cv=5,
                                                                                                            scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared,   housing_labels)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def randomFAlgorithm(winePdf):
    x = winePdf.drop(["quality"], axis = 1).values

    y = winePdf["quality"].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.25)
    rfModel = RandomForestClassifier(criterion = "entropy", n_estimators = 200, max_depth=90,
                                     max_features=3, min_samples_leaf=3, min_samples_split=10)
    rfModel.fit(x_train, y_train)
    randomForrestFile = open('resultFiles/randomForrestResult.txt', 'w')
    randomForrestFile.write("RESULT OF RANDOM FORREST ALGORITHM\n\n")
    randomForrestFile.write("Accuracy score for training set: "+str(round(rfModel.score(x_train,y_train), 3)*100)+"%\n")
    randomForrestFile.write("Accuracy score for test set: "+str(round(rfModel.score(x_test,y_test), 3)*100)+"%")
    predictionOfModel = rfModel.predict(x_test);
    target_names = ["Bad", "Good"]
    randomForrestFile.write("\n\nClassification report:\n\n "+str(classification_report(y_test, predictionOfModel, target_names = target_names)))
    randomForrestFile.close();
    #hyperparamether(x_train, y_train)
    
def hyperparamether(x_train, y_train):
    param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
    }
    # Create a based model
    rf = RandomForestClassifier()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
    best_model = grid_search.fit(x_train,y_train)
    print("Best max depth: ", best_model.best_estimator_.get_params()['max_depth'])
    print("Best max features: ", best_model.best_estimator_.get_params()['max_features'])
    print("Best min samples leaf: ", best_model.best_estimator_.get_params()['min_samples_leaf'])
    print("Best min samples split: ", best_model.best_estimator_.get_params()['min_samples_split'])
    print("Best n estimators: ", best_model.best_estimator_.get_params()['n_estimators'])


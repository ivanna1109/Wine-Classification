from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def knnClassifier(winePdf):
    x = winePdf.drop(["quality"], axis = 1).values
    y = winePdf["quality"].values
    scaler = MinMaxScaler(feature_range=(0,1))
    x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=4, test_size=0.25)
    scaler.fit_transform(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    knnModel = KNeighborsClassifier()
    knnModel.fit(x_train, y_train)
    y_pred = knnModel.predict(x_test)
    knnClassifierFile = open('resultFiles/knnClassifierResult.txt', 'w')
    knnClassifierFile.write("RESULT OF KNN ALGORITHM\n\n")
    knnClassifierFile.write("Classification report for KNN:\n\n"+classification_report(y_test, y_pred)+"\n")
    scoreForTraining = round(knnModel.score(x_train, y_train)*100, 2)
    scoreForTest = round(knnModel.score(x_test, y_test)*100, 2)
    knnClassifierFile.write(format("-"*55))
    knnClassifierFile.write("\nAccuraccy score for training set: " f"{scoreForTraining}"+"%\n")
    knnClassifierFile.write("Accuraccy score for test set: " f"{scoreForTest}"+"%\n")
    knnClassifierFile.write(format("-"*55))
    knnClassifierFile.write("\nKnn with tuned parameter\n ")
    hyperParamether(x,y, x_train, y_train, x_test, y_test, knnClassifierFile)
    #print("Best leaf size and n_neighbour and p:\n")
    #testingTheBest(x,y)

def hyperParamether(x,y, x_train, y_train, x_test, y_test,knnClassifierFile):
    
    knn_2 = KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='minkowski',
                                 metric_params=None, n_jobs=1, n_neighbors=28, p=1,
                                 weights='uniform')#Use GridSearch
    knn_2.fit(x_train, y_train)
    y_pred = knn_2.predict(x_test)
    knnClassifierFile.write("\nClassification report for KNN:\n\n"+classification_report(y_test, y_pred)+"\n")
    scoreForTraining = round(knn_2.score(x_train, y_train)*100, 2)
    scoreForTest = round(knn_2.score(x_test, y_test)*100, 2)
    knnClassifierFile.write(format("-"*55))
    knnClassifierFile.write("\nAccuraccy score for training set: " f"{scoreForTraining}"+"%\n")
    knnClassifierFile.write("Accuraccy score for test set: " f"{scoreForTest}"+"%\n")
    knnClassifierFile.close()
    
def testingTheBest(x,y):
    leaf_size = list(range(1,50))
    n_neighbors = list(range(1,30))
    p=[1,2]#Convert to dictionary
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    knn_2= KNeighborsClassifier()
    clf = GridSearchCV(knn_2, hyperparameters, cv=10)#Fit the model
    best_model = clf.fit(x,y)#Print The value of best Hyperparameters
    print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
    print('Best p:', best_model.best_estimator_.get_params()['p'])
    print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
    
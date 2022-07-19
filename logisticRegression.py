import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression # for Logistic Regression Algorithm

def logisticRegression(winePdf):
    winePdf["quality_category"] = winePdf["quality"].astype("category").cat.codes
    x = winePdf.iloc[:,:13]
    y = winePdf["quality_category"]
    x.drop(["quality","quality_category"],axis=1,inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=42)
    scaler=MinMaxScaler(feature_range=(0,1))
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    logisticModel = LogisticRegression(solver='liblinear')
    logisticModel.fit(x_train,y_train)
    logisticRegressionFile = open('resultFiles/logisticRegressionResult.txt', 'w')
    logisticRegressionFile.write("RESULT OF LOGISTIC REGRESSION ALGORITHM\n\n")
    logisticRegressionFile.write("Accuracy score for training set: "+str(round(logisticModel.score(x_train,y_train), 3)*100)+ "%\n")
    logisticRegressionFile.write("Accuracy score for test set: "+str(round(logisticModel.score(x_test,y_test), 3)*100)+"%\n")
    logisticRegressionFile.write("Cross validation score for this model: "+str(cross_val_score(logisticModel, x, y, cv=5))+"\n")
    logisticRegressionFile.write("Accuracy score for training set after CV: "+str(round(logisticModel.score(x_train,y_train), 3)*100)+ "%\n")
    predictionOfModel = logisticModel.predict(x_test)
    matrix = confusion_matrix(y_test,predictionOfModel)
    sns.heatmap(matrix, annot = True)
    target_names = ['Bad', 'Good']
    logisticRegressionFile.write(format("-"*55))
    logisticRegressionFile.write("\nClassification report:\n\n "+str(classification_report(y_test, predictionOfModel, target_names = target_names)))
    logisticRegressionFile.close()


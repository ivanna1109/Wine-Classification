from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

def svmAlgorithm(winePdf):
    x = winePdf.drop(["quality"], axis = 1).values
    y = winePdf["quality"].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    linearSVCModel = LinearSVC(max_iter=12000, dual=False, verbose=0)
    linearSVCModel.fit(x_train, y_train)
    linearSVMFile = open('resultFiles/linearSVMResult.txt', 'w')
    linearSVMFile.write("RESULT OF LINEAR SUPPORT VECTOR MACHINE ALGORITHM\n\n")
    cv_scores = cross_val_score(linearSVCModel, x_train, y_train, cv=10)*100
    linearSVMFile.write("Cross validation average test score: "+str(round(cv_scores.mean(), 2))+"%\n")
    linearSVMFile.write(format("-"*55))
    y_pred = linearSVCModel.predict(x_test)
    linearSVMFile.write(format("-"*55))
    linearSVMFile.write("\nLinear Support Vector Machine classification report: \n\n")
    cr = classification_report(y_test, y_pred)
    linearSVMFile.write(str(cr))
    scoreForTraining = round(linearSVCModel.score(x_train, y_train)*100, 2)
    scoreForTest = round(linearSVCModel.score(x_test, y_test)*100, 2)
    linearSVMFile.write(format("-"*55))
    linearSVMFile.write("\nLinear Support Vector Machine algorithm score \n\n")
    linearSVMFile.write("Accuraccy score for training set: " f"{scoreForTraining}"+"%\n")
    linearSVMFile.write("Accuraccy score for test set: " f"{scoreForTest}"+"%")
    linearSVMFile.close()





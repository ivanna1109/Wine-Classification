from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


def decisionTree(winePdf):
    decisionTreeModel = DecisionTreeClassifier()
    winePdf['quality_category'] = winePdf['quality']
    winePdf.drop('quality',axis=1,inplace=True)
    x = winePdf.drop('quality_category',axis=1)
    y = winePdf["quality_category"]
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.25,random_state=42)
    decisionTreeModel.fit(x_train, y_train)
    decisionTreeFile = open('resultFiles/decisionTreeResult.txt', 'w')
    decisionTreeFile.write("RESULT OF DECISION TREE ALGORITHM\n\n")
    decisionTreeFile.write("Accuracy score for training set: "+str(round(decisionTreeModel.score(x_train,y_train), 3)*100)+ "%\n")
    decisionTreeFile.write("Accuracy score for test set: "+str(round(decisionTreeModel.score(x_test,y_test), 2)*100)+ "%\n")
    prediction = decisionTreeModel.predict(x_test)
    target_names = ['Bad', 'Good']
    decisionTreeFile.write(str(format("-"*55)))
    decisionTreeFile.write("\nClassification report:\n\n\n"+str(classification_report(y_test, prediction, target_names=target_names)+"\n\n"))
    
    #improvedDecisionTree
    decisionTreeFile.write(str(format("-"*55)))
    decisionTreeFile.write("\n\nImproved decision tree by hyperparameter \n\n")
    improvingTree(x, y, decisionTreeFile)
    
def improvingTree(x, y, decisionTreeFile):
    #experimentForMaxDepth(x, y)
    #experimentForMaxLeafs(x,y)
    decisionTreeImproved = DecisionTreeClassifier(max_depth=13,max_leaf_nodes=50)
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.25,random_state=0)
    decisionTreeImproved.fit(x_train,y_train)
    scoreForTraining = round(decisionTreeImproved.score(x_train,y_train)*100, 2)
    scoreForTest = round(decisionTreeImproved.score(x_test,y_test)*100, 2)
    decisionTreeFile.write("Accuraccy score for training set: "f"{scoreForTraining}"+"%\n")
    decisionTreeFile.write("Accuraccy score for test set: " f"{scoreForTest}"+"%")
    decisionTreeFile.close()
    
def experimentForMaxDepth(x,y):
    x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
    for max_d in range(1,21):
        model = DecisionTreeClassifier(max_depth=max_d, random_state=42)
        model.fit(x_train, y_train)
        print('The Training Accuracy for max_depth {} is:'.format(max_d), round(model.score(x_train, y_train)*100,2))
        print('The Validation Accuracy for max_depth {} is:'.format(max_d), round(model.score(x_val,y_val)*100,2))
        print('')

def experimentForMaxLeafs(x,y):
    x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
    array = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    for max_d in array:
        model = DecisionTreeClassifier(max_leaf_nodes=max_d, random_state=42)
        model.fit(x_train, y_train)
        print('The Training Accuracy for max_number_leafs {} is:'.format(max_d), round(model.score(x_train, y_train)*100,2))
        print('The Validation Accuracy for max_number_leafs {} is:'.format(max_d), round(model.score(x_val,y_val)*100,2))
        print('')


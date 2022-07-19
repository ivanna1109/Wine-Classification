import pandas as pd
import graphicRepresentation as graphics
import logisticRegression as lr
import decisionTreeClassifier as dt
import knnClassifier as knn
import pca as pca
import linearSVM as svm
import randomForrest as rf

def readFile():
    pdf = pd.read_csv('D:/Fakultet/4. godina/Letnji semestar/AI/MLProjectIM2218/WineClassificationIM/wine.csv')
    dataFile = open('resultFiles/dataFile.txt', 'w')
    dataFile.write(str(pdf.head()));
    dataFile.close()
    return pdf;

def main():
    winePdf = readFile()
    #graphics.pieGraph(winePdf);
    #graphics.stubGraph(winePdf)
    
    #graphics.relationshipOfColumnAndTarget(winePdf)
    
    #distribucija po parametrima
    #graphics.distributionByParams(winePdf, 'pH', 'citric acid')

    #matrica korelisanosti parametara dataseta
    #graphics.graphCorellation(winePdf)
    
    #distribution plot for all columns
    #graphics.distributionPlot(winePdf)
    
    winePdf['quality'].replace({'bad': 0 , 'good': 1}, inplace=True)
    
    #PCA
    #pca.principalCompAnalysis(winePdf)

    #logistic regression
    #lr.logisticRegression(winePdf)
    
    #decisiontree and improving
    #dt.decisionTree(winePdf)
    
    #KNN algorithm
    #knn.knnClassifier(winePdf)
    
    #Linear SVC
    #svm.svmAlgorithm(winePdf)
    
    #random forrest
    rf.randomFAlgorithm(winePdf)
    
    
    print("Check the results of algorithms in folder resultFiles.")
    

if __name__ == "__main__":
    main() 

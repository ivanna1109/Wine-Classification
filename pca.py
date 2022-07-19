import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def principalCompAnalysis(winePdf):
    y = winePdf['quality'].values
    y = y.reshape(-1,1)
    x = winePdf.drop(['quality'],axis = 1)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.33,random_state=100)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    scaler=StandardScaler()
    x_train, y_train = scaler.fit_transform(x_train), scaler.fit_transform(y_train)
    pca = PCA(n_components=2)
    x_pca = scaler.fit_transform(x)
    model = pca.fit_transform(x_pca)
    m_df= pd.DataFrame(data = model, columns= ['PC 1', 'PC 2'])
    winePdf['quality'].replace({0:'bad', 1:'good'}, inplace=True)
    final_result = pd.concat([m_df, winePdf[['quality']]], axis=1)
    pcaFile = open("resultFiles/pcaResult.txt", "w")
    pcaFile.write("RESULT OF PRINCIPAL COMPONENT ANALYSIS\n\n")
    pcaFile.write(str(final_result.head()))
    final_result['quality'].replace({'bad': 0 , 'good': 1}, inplace=True)
    graphic(final_result)

def graphic(final_result):
    figure = plt.figure(figsize=(8,8))
    ax = figure.add_subplot(1,1,1)
    ax.set_xlabel('PC 1',fontsize = 15)
    ax.set_ylabel('PC 2',fontsize = 15)
    targets = [0,1]
    colors=['r','g']
    for target , color in zip(targets, colors):
        indicesToKeep = final_result['quality'] == target
        ax.scatter(final_result.loc[indicesToKeep,'PC 1']
             ,final_result.loc[indicesToKeep,'PC 2']
             ,c = color
             ,s = 50)
    targets[0] = 'bad'
    targets[1] = 'good'
    ax.legend(targets)
    ax.grid()


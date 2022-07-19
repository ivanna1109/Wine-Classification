
import matplotlib.pyplot as plt
import seaborn as sns

def stubGraph(df):
    plt.xlabel("Good or Bad")
    plt.ylabel("Count")
    plt.title("Quality")
    plt.figure(num="Stub chart")
    sns.countplot(x=df.quality)

#iscrtan grafik pite da vidimo da je data set uglavnom balansiran
def pieGraph(winePdf):
    plt.figure(figsize=(40,25))
    plt.subplots_adjust(left=0, bottom=0.1, right=1, top=0.9, wspace=0.5, hspace=0.8)
    plt.subplot(111)
    plt.title('Percentage of good and bad quality wine',fontsize = 20)
    plt.figure(num="Pie chart")
    winePdf['quality'].value_counts().plot.pie(autopct="%1.1f%%")
    
def relationshipOfColumnAndTarget(df):
    for col in df.drop("quality", axis=1).columns:
        plt.figure(figsize=(10,8))
        sns.barplot(x=df["quality"], y=df[col])
        plt.title(f"{col} and quality", size=15)
        plt.show()

#prikazuje koliko je su parametri po parametrima dobro korelisani    
def graphCorellation(winePdf):
    corr = winePdf.corr()
    plt.figure(figsize=(10,8)) 
    sns.heatmap(corr, cmap='coolwarm', annot=True)
  
def distributionPlot(df):
    fig = plt.figure(figsize = (15,20))
    ax = fig.gca()
    df.hist(ax = ax)
    
def distributionByParams(winePdf, param1, param2):
    sns.scatterplot(x = param1, y = param2, hue = 'quality', data=winePdf)

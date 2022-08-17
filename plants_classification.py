from matplotlib import markers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


def separation(data):
    if data==1:
        X=np.array(iris.drop(columns='Species'))
    elif data==2: #Sepal
        X=np.array(iris.drop(columns=['Species','PetalLengthCm','PetalWidthCm']))
    elif data==3: #Petal
        X=np.array(iris.drop(columns=['Species','SepalLengthCm','SepalWidthCm']))
    return X


def superalgorithm(alg):
    if alg==1:
        algorithm=LogisticRegression()
        txt='Logistic Regression Score: {}'
    elif alg==2:
        algorithm=SVC()
        txt='SVC(Support Vector Classification) Score : {}'  
    elif alg==3:
        algorithm=KNeighborsClassifier()
        txt='Neighbors Classifier Score: {}'
    elif alg==4:
        algorithm=DecisionTreeClassifier()
        txt='Decision Tree Classifier Score : {}' 
        #Score: The mean accuracy on the given test data and labels

    algorithm.fit(X_train,y_train)
    Y_pred=algorithm.predict(X_test)
    print(txt.format(algorithm.score(X_train,y_train)))

    #CONFUSION MATRIX
    labels=['Iris-setosa','Iris-versicolor','Iris-virginica']
    cm=confusion_matrix(Y_pred,y_test,labels=labels)
    sns.heatmap(cm,annot=True,xticklabels=labels,yticklabels=labels,linewidth=5,cmap='coolwarm')
    plt.title('Confusion matrix')
    plt.xlabel('True Class')
    plt.ylabel('Predict Class')
    plt.show()


#Dataset info
iris=pd.read_csv('Iris.csv')
iris=iris.drop('Id',axis=1)
print('\nCabecera del dataset:')
print(iris.head(3))
print('\nInfo del dataset:')
print(iris.info())
print('\nDescripción del dataset:')
print(iris.describe())
print('\nDistribución de las especies de Iris:')
print(iris.groupby('Species').size())

#Sepal plot
fig=iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',
y='SepalWidthCm',color='blue',marker='o',label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',
y='SepalWidthCm',color='green',marker='o',label='Versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',
y='SepalWidthCm',color='red',marker='o',label='Virginica',ax=fig)
fig.set_title('SepalLenght vs SepalWidth')
plt.legend(loc='upper right',bbox_to_anchor=(1.3,1))
plt.show()
#Petal plot

fig=iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',
y='PetalWidthCm',color='blue',marker='o',label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',
y='PetalWidthCm',color='green',marker='o',label='Versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',
y='PetalWidthCm',color='red',marker='o',label='Virginica',ax=fig)
fig.set_title('PetalLenght vs PetalWidth')
plt.legend(loc='upper right',bbox_to_anchor=(1.3,1))
plt.show()

#Separación de datos
data=int(input("""What data would do you like use?
    1- Sepal&Petal
    2- Sepal
    3- Petal
"""))
X=separation(data)
y=np.array(iris['Species'])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2) #AQUÍ SE CAMBIA LA PROPORCIÓN
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))

#Elección del algoritmo
alg=int(input("""
What algorithm would do you like use?
    1- Logistic Regresion
    2- SVC(Máquinas de Vectores de Soporte)
    3- KNeighbors Classifier
    4- Decision Tree Classifier"""))
superalgorithm(alg)
# Aqui va toda la parte de Machine Learning (modelos)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.cluster import MeanShift, MiniBatchKMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from utils import Utils

class Models:

    def __init__(self): # Este es un constructor, es la primera funcion que se ejecuta cuando llamo una clase
        self.reg = {
            'SVR' : SVR(),
            'GRADIENT' : GradientBoostingRegressor()
        }

        self.params = {
            'SVR' : {
                'kernel' : ['linear', 'poly', 'rbf'],
                'gamma' : ['auto','scale'],
                'C' : [1,5,10]
            }, 'GRADIENT' :{
                'loss' : ['squared_error','huber'],
                'learning_rate' : [0.01, 0.05, 0.1]
            }
        }
    
    def grid_trainning(self,X,y):

        best_score = 999
        best_model = None

        for name, reg in self.reg.items():

            grid_reg = GridSearchCV(reg, self.params[name], cv=3).fit(X,y.values.ravel())
            score = np.abs(grid_reg.best_score_)

            if score < best_score:
                best_score = score
                best_model = grid_reg.best_estimator_
        
        utils = Utils()
        utils.model_export(best_model,best_score)


    def regularization(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

        modelLinear = LinearRegression().fit(X_train,y_train)
        y_predict_linear = modelLinear.predict(X_test)

        modelLasso = Lasso(alpha=0.02).fit(X_train,y_train)
        y_predict_lasso = modelLasso.predict(X_test)

        modelRidge = Lasso(alpha=0.02).fit(X_train,y_train)
        y_predict_ridge= modelRidge.predict(X_test)

        linear_loss = mean_squared_error(y_test,y_predict_linear)
        print("Linear loss:",linear_loss)
        lasso_loss = mean_squared_error(y_test,y_predict_lasso)
        print("Lasso loss:",lasso_loss)
        Ridge_loss = mean_squared_error(y_test,y_predict_ridge)
        print("Ridge loss:",Ridge_loss)

        print("="*32)
        print("Coef lasso",modelLasso.coef_)
        print("="*32)
        print("Coef ridge",modelRidge.coef_)
        print("="*32)
        print("Coef linear",modelLinear.coef_)  

    def meanshift(self,X,y):
        
        dataset = pd.read_csv('./in/tetuan_power.csv')


        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X) 

        meanshift = MeanShift(bandwidth=0.5)
        meanshift.fit(X_scaled)  
        cluster_labels =meanshift.labels_
        # print(max(cluster_labels))

        dataset['cluster'] = cluster_labels
        print(dataset.head(5))


        pca = PCA(n_components=2)  
        pca_data = pca.transform(X_scaled)
            
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels)
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c='black', s=200, marker='x', label='Centers') 
        plt.title('MeanShift Clustering with PCA Visualization')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()  


    def kmeans(Self,X,y):
        dataset = pd.read_csv('./in/tetuan_power.csv')

        kmeans = MiniBatchKMeans(n_clusters = 1,batch_size=8).fit(X)
        dataset['group'] = kmeans.predict(X)

        sns.scatterplot(data=dataset, x="DateTime", y="Zone1", hue="group",palette="deep")
        plt.show()

    def treeClassifier(self,X,y,details):
        model = DecisionTreeClassifier().fit(X,y)

        predictions = model.predict(details)
        print(predictions)
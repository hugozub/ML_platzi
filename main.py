from utils import Utils
from models import Models

if __name__=="__main__":

    utils = Utils()
    models = Models() # Con este comando se ejecuta el constructor

    data = utils.load_from_csv('./in/music.csv')
    X = data.drop(['genre'],axis=1)        
    y = data['genre']
    
    # Se manda la edad y el sexo de las personas que se queire predecir,
    # Se busca saber que genero musica le gusta a personas con  la edad de 21 y 28 años para hombre y 22 de mujer
    details = [[21,1],[22,0],[28,1]]

    models.treeClassifier(X,y,details)


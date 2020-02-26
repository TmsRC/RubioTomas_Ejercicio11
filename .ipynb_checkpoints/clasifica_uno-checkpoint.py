import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score


numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
print(np.shape(imagenes), n_imagenes)


data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
print(np.shape(data))


scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


numero = 1
dd = y_train==numero
cov = np.cov(x_train[dd].T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]



proyeccion_prueba = np.matmul(x_train,vectores)
proyeccion_test = np.matmul(x_test,vectores)
discriminante = LinearDiscriminantAnalysis()



def darF1(numComponentes):    
    discriminante.fit(proyeccion_prueba[:,:numComponentes],1.0*(y_train==1))
    prediccion_train = discriminante.predict(proyeccion_prueba[:,:numComponentes])
    prediccion_test = discriminante.predict(proyeccion_test[:,:numComponentes])
    scoreTrain,scoreTest = f1_score(1.0*(y_train==1),prediccion_train),f1_score(1.0*(y_test==1),prediccion_test)
    return scoreTrain,scoreTest

def darF1_otros(numComponentes):    
    discriminante.fit(proyeccion_prueba[:,:numComponentes],1.0*(y_train!=1))
    prediccion_train = discriminante.predict(proyeccion_prueba[:,:numComponentes])
    prediccion_test = discriminante.predict(proyeccion_test[:,:numComponentes])
    scoreTrain,scoreTest = f1_score(1.0*(y_train!=1),prediccion_train),f1_score(1.0*(y_test!=1),prediccion_test)
    return scoreTrain,scoreTest


fig,ax = plt.subplots(1,2)
for i in range(1,61):
    f1_train,f1_test = darF1(i)
    ax[0].scatter(i,f1_train,c='r')
    ax[0].scatter(i,f1_test,c='b')  
    ax[0].set_xlabel('Número de componentes')
    ax[0].set_ylabel('F1 Score')
    ax[0].set_title('Clasificación UNOS')

for i in range(1,61):
    f1_train,f1_test = darF1_otros(i)
    ax[1].scatter(i,f1_train,c='r')
    ax[1].scatter(i,f1_test,c='b')
    ax[1].set_xlabel('Número de componentes')
    ax[1].set_ylabel('F1 Score')
    ax[1].set_title('Clasificación OTROS')
    

fig.tight_layout()
fig.savefig('grafica_F1.png')
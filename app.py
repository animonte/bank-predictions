import gradio as gr
from gradio import components
import pandas as pd 
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import MultinomialNB


# Lectura del CSV desde un data set
dataset = load_dataset("animonte/bank-state-data")
df = pd.DataFrame(dataset["train"])

# Primera eliminación de columnas
df = df.drop(['Unnamed: 0','RowNumber','Surname','CustomerId'], axis=1)

# Transformación de variables categóricas a numéricas
Gender={'Female':0,'Male':1}
Geography = {'Texas':1,'California':2,'Alabama':3}

df['Gender'].replace(Gender, inplace=True)
df['Geography'].replace(Geography, inplace=True)

# Imputación de nulos
df = df.apply(lambda x:x.fillna(x.median()))

# Imputación de outliers
def imputar_outliers(df, nombre_columna):
    Q3 = np.percentile(df[nombre_columna], 75)
    Q1 = np.percentile(df[nombre_columna], 25)
    RI = Q3 - Q1

    limite_superior = Q3 + 1.5 * RI
    limite_inferior = Q1 - 1.5 * RI

    df[nombre_columna] = np.where(df[nombre_columna] > limite_superior, 
                                  np.percentile(df[nombre_columna], 97),
                                  df[nombre_columna])

    df[nombre_columna] = np.where(df[nombre_columna] < limite_inferior, 
                                  np.percentile(df[nombre_columna], 5),
                                  df[nombre_columna])
    return df[nombre_columna]

variables_outlier = ['Age','Tenure']

for col in variables_outlier:
    df[col] = imputar_outliers(df, col)

    #print(df)

# Fin Imputación de outliers

# Algoritmo Regresión Logística para modelo predictivo

# Variables incluidas en el modelo
pred_labels = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']

X = df[pred_labels]
y = df['Exited']

# Subdividimos el dataset
kfold = KFold(n_splits=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Modelo Árbol de Decicisón
modelo_ajusta_train = tree.DecisionTreeClassifier(max_depth = 4, criterion = 'gini').fit(X_train, y_train)
modelo_entrenado = modelo_ajusta_train.predict_proba(X_test)

# Modelo Naive Bayes
#Creamos un objeto de Naive Bayes Multinomial
modelo_naives = MultinomialNB()

#Entrenamos el modelo con los datos de entrenamiento
modelo_naives.fit(X_train,y_train)

# Interfaz grafica
def predict(Score, Age, Balance, Salary):
    
    inputs = [Score, Age, Balance, Salary]

    probabilidad_de_que_sea_1 = modelo_ajusta_train.predict_proba([inputs])[0][1]

    if probabilidad_de_que_sea_1 > 0.08:
       prediccion_arbol = "Abandona el banco"
    else:
        prediccion_arbol = "Se queda en el banco."

    predicciones_naives = modelo_naives.predict([inputs])

    if predicciones_naives == 0:
        resultado_naives = "Se queda en el banco."
    else:
        resultado_naives = "Abandona el banco"

    return prediccion_arbol, resultado_naives

output_tree = components.Textbox(label='Prueba con el modelo Tree con una sensibilidad del 0.08')
output_naives = components.Textbox(label='Prueba con el modelo Naives')


demo = gr.Interface(
    fn=predict, 
    inputs=[gr.Slider(350, 850), "number","number","number"], 
    outputs=[output_tree, output_naives],
    allow_flagging="never"
    )

demo.launch()

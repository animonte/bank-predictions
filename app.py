# import gradio as gr
import pandas as pd 
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
# df['target'] = pd.Series(dataset.target)

# print(df)

# def greet(name):
#     return "Hello " + name

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")

# demo.launch()

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

# Entrenamos y testeamos
model_logistic = LogisticRegression()
model_logistic.fit(X_train,y_train)
predicted = model_logistic.predict(X_test)

# Hacemos las predicciones con los datos en test
result_logistic_score = cross_val_score(model_logistic, X, y, cv=kfold)
print("SCORE: ", result_logistic_score.mean())
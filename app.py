# import gradio as gr
import pandas as pd 
import numpy as np
from datasets import load_dataset

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

    print(df)

# Fin Imputación de outliers
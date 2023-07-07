# import gradio as gr
import pandas as pd 
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
 
print(df)

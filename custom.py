import pandas as pd
import lightgbm as lgb
import numpy
import os
import io
import joblib
from io import StringIO

def init(code_dir):
     global g_code_dir
     g_code_dir = code_dir

def load_model(code_dir):
    #model_path = "lightgbm.pkl"
    #model = joblib.load(os.path.join(code_dir, model_path)) 
    global model
    filename = '/lightgbm.pkl'
    model = joblib.load(open(code_dir+filename, 'rb'))
    return model

def read_input_data(input_binary_data): 
     # load pipeline.pkl for data transformation
     # pipeline_path="/pipeline.pkl"
     # pipeline=joblib.load(open(g_code_dir+pipeline_path, 'rb'))
     #Apply transformation on data
     #data = pipeline.transform(input_binary_data)
     return pd.read_csv(io.BytesIO(input_binary_data)) 

def score(data,model,**kwargs):
     res = model.predict(data)
     dataset = pd.DataFrame(res)
     return pd.DataFrame(dataset)

def transform(data, model):
    data = data.fillna(0)
    return data

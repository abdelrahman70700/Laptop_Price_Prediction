import streamlit as st
import pickle
import numpy as np
import pandas as pd

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))

df = pickle.load(open('laptops.pkl','rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('TypeName',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop',min_value=1.0)

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size',min_value=1.0)

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['Cpu_Brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu_Brand'].unique())

os = st.selectbox('OS',df['OpSys'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    # st.write(int(ram))
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,os,weight,touchscreen,ips,ppi,gpu,cpu,hdd,ssd])
    query = query.reshape(1,12)
    query = pd.DataFrame(query, columns=["Company", "TypeName", "Ram", "OpSys", "Weight", "Touchscreen", "Ips", "ppi",
                                         "Gpu_Brand", "Cpu_Brand","HDD","SSD"])
    query["Ram"] = query["Ram"].astype(int)
    query["Weight"] = query["Weight"].astype(float)
    query["Touchscreen"] = query["Touchscreen"].astype(int)
    query["Ips"] = query["Ips"].astype(int)
    query["ppi"] = query["ppi"].astype(float)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)))))


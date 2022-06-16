from turtle import clear
import streamlit as st 
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 
st.sidebar.write("UNIVERSITY OF OXFORD")
st.sidebar.write("""A city of dreams and a city of science. 
                  A city where the world comes together to 
                  create the future, surrounded by the past.  
                  A city that for centuries has attracted 
                  those in search of excellence and answers, 
                  now guarding a millennium’s worth of stories!""")
st.sidebar.write(" ")
st.sidebar.write(" ANGGOTA ")
st.sidebar.write("1. AKHIL NUR RIYADI - 20SA1280 ")
st.sidebar.write("2. SIDIQ NUR FAHREZA - 20SA1209 ")
st.sidebar.write("3. ADITYA RAMADHAN - 20SA1033 ")
st.sidebar.write("4. MUHAMAD SOLEH - 20SA1082 ")
st.sidebar.write("5. ZHAFRAN AFIF NURDIYANSAH - 20SA1067 ")

st.image('./image1.png')
st.write("# ")
st.write("# ")
st.write("# ")
st.write("# ")
st.write("""
    # Prediksi Peluang UNIVERSITY OF OXFORD""")
st.write( """
    ## Persyaratan Masuk
    1. GRE Score: Merupakan Score Test Untuk Masuk Program S2 (0 - 615) Bersifat Continous
    2. TOEFL Score: Score Kemampuan TOEFL (0 - 140) Bersifat Continous
    3. University Rating: Rating Universitas (0 - 5) Bersifat Ordinal
    4. Kekuatan Surat Rekomendasi (0 - 5) Bersifat Ordinal
    5. GPA Sewaktu Undergraduate (0 - 10) Bersifat Continous
    6. Pengalaman Riset (0 : tidak ada, 1 : ada) Bersifat Nominal
    
    7. Peluang Diterima (0 - 1) Merupakan Dependent Variable
""")

st.write("# ")
st.write("# ")
st.write("""
    ## Overview Data
""")

myData = pd.read_csv('data1.csv')

st.dataframe(myData)

st.write("# ")
st.write("# ")
st.write("""
    ## Deskripsi Data
""")

st.dataframe(myData.describe())

# Preproccessing Data
st.write("# ")
st.write("# ")
st.write("""
    ## Dilakukan Preprocessing Data dimana Fitur dan Labelnya akan Dipisah
""")

# Memisahkan Label Dan Fitur 
X = myData.iloc[:, 1:-1].values
y = myData.iloc[:, -1].values



st.write("## Input Data X",X)
st.write("## Label Data y",y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)



from sklearn.preprocessing import StandardScaler 

ss_train_test = StandardScaler()


X_train_ss_scaled = ss_train_test.fit_transform(X_train)
X_test_ss_scaled = ss_train_test.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

l_regressor_ss = LinearRegression()
l_regressor_ss.fit(X_train_ss_scaled, y_train)
y_pred_l_reg_ss = l_regressor_ss.predict(X_test_ss_scaled)

st.write("# ")
st.write("# ")
st.write("Dengan Menggunakan Multiple Linear Regression Diperoleh Skor Untuk Data Test")
st.write(r2_score(y_test, y_pred_l_reg_ss))

st.write("# ")
st.write("# ")
st.write("# Silahkan Masukan Nilai Test Anda")
form = st.form(key='my-form')
inputGRE = form.number_input("Masukan GRE Score: ", 0)
inputTOEFL = form.number_input("Masukan TOEFL Score: ", 0)
inputUnivRating = form.number_input("Masukan Rating Univ: ", 0)
inputSOP = form.number_input("Masukan Kekuatan SOP: ", 0)
inputLOR = form.number_input("Masukan Kekuatan LOR: ", 0)
inputCGPA = form.number_input("Masukan CGPA: ", 0)
inputResearch = form.number_input("Pengalaman Researc, 1 Jika Pernah Riset, 0 Jika Tidak", 0)
submit = form.form_submit_button('Submit')
    
completeData = np.array([inputGRE, inputTOEFL, inputUnivRating, 
                        inputSOP, inputLOR, inputCGPA, inputResearch]).reshape(1, -1)
scaledData = ss_train_test.transform(completeData)
                        
st.write('Tekan Submit Untuk Melihat Prediksi Peluang S2 Anda')

if submit:
    prediction = l_regressor_ss.predict(scaledData)
    if prediction > 1 :
        result = 1
    elif prediction < 0 :
        result = 0
    else :
        result = prediction[0]
    st.write(result*100, "Percent")

 




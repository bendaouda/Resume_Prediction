import streamlit as st
import os
import json
import os
import re
import numpy as np
import pandas as pd
import PyPDF2
import nltk
from nltk import SnowballStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
#Pour traiter tout ce qui contient du html dans le texte
from bs4 import BeautifulSoup
# Partitionnement du jeu de données
from sklearn.model_selection import train_test_split
# Graphiques
from matplotlib import pyplot as plt
import seaborn as sns
#For OCR
from PIL import Image as img
import pytesseract as PT  
import sys  
from pdf2image import convert_from_path as CFP
import joblib
from pathlib import Path
import functions.functions as f
import shutil

current_dir = Path(__file__).parent.parent if "__file__" in locals() else Path.cwd()
path_tmp = current_dir / "tmp"
path_tmp=str(path_tmp)+'\\'
path_static = current_dir / "static" 
path_static=str(path_static)+'\\'

st.set_page_config(
    page_title="NLP PROJECT",
    page_icon="🖥️",
)

#Remove the "Made with Streamlit" at footer and remove the menu of the top right
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.markdown("### ADD CV TO ITS PREDICTED CATEGORY ")

def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

for filename in os.listdir(path_tmp):
        file_path = os.path.join(path_tmp, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

with st.form("my_form"):
    pathname = st.text_input('Path to Datasets', "C:/Users/soule/OneDrive/Documents/M2 BI/NPL/PROJET FINAL/Projet final_Text Mining 2022/Jeu_de_donnees_bis/cv_dataset/")
    uploaded_files = st.file_uploader("Choose your PDF file",type="PDF",accept_multiple_files=True)
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    tableau=[]
    for uploaded_file in uploaded_files:
        # Save uploaded file to 'F:/tmp' folder.
        save_folder = path_tmp
        save_path = Path(save_folder, uploaded_file.name)
        with open(save_path, mode='wb') as w:
            w.write(uploaded_file.getvalue())
        chemin=path_tmp
        cvs_output=f.transform_new_CV(chemin +  uploaded_file.name)
        data_new=pd.DataFrame(data=cvs_output,columns=["cv"])
        # Nettoyage
        #Liste des stopwords de la langue anglaise
        english_stops = list(set(stopwords.words("english")))
        #On récupère les stopwords liés au jargon des cvs
        with open(path_static+"specific_stopwords.txt") as fichier:
            specific_stops = [stop.strip() for stop in fichier.readlines()]
        #Liste des stopwords
        stop_words = english_stops+specific_stops
        #Mauvais caractères
        Mauvais_caract = re.compile('[^0-9a-z \'#+_]')
        data_new.cv = data_new.cv.apply(lambda document: f.clean(document, stop_words, Mauvais_caract))
        #prediction
        cv_classifier = joblib.load(path_static+'cv_classifier.pkl') 
        X_train_vec = joblib.load(path_static+'X_train_vec.pkl')
        cv_new=data_new.iloc[0,0]
        new_cv_trans = X_train_vec.transform([cv_new])
        #print(new_cv_trans)
        # On prédit sa classe
        pred = cv_classifier.predict(new_cv_trans)
        tableau.append([uploaded_file.name,pred[0]])
        #print(pathname+pred[0])
        shutil.copy(path_tmp+uploaded_file.name, pathname+pred[0])
        ActiveFile=path_tmp+uploaded_file.name
        #os.remove(path_tmp+uploaded_file.name)
    df=pd.DataFrame(tableau,columns=("resume Name","Predicted Categorie"))
    st.dataframe(df)
    

    #st.markdown("###### PREDICTED CATEGORIE FOR : "+uploaded_file.name+" is : "+pred[0])



































#st.markdown("")
#st.markdown("### About this app")
#with st.expander("ℹ️ - About this app", expanded=True):
#   st.write(
 #   ""    
  #  ""
   # ) 
        
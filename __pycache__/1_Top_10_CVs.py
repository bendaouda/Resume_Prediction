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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import functions.functions as f
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
path_tmp = current_dir / "tmp/"
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


st.markdown("### TOP 10 CV BY CATEGORY")

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


with st.form("my_form"):
    pathname = st.text_input('Path to Datasets', "C:/Users/soule/OneDrive/Documents/M2 BI/NPL/PROJET FINAL/Projet final_Text Mining 2022/Jeu_de_donnees_bis/cv_dataset")
    pathname = pathname.replace("\\","/")
    submit_button = st.form_submit_button(label="Submit")
if submit_button:
    #Lecture de tous les CV
    data_bis = f.lit_cvs_type(pathname)
    cv_type=pd.DataFrame(data=data_bis, columns=["cv", "categorie","_type"])
    cv_type["ids"]=[i for i in range(0,cv_type.shape[0])]

    #Liste des stopwords de la langue anglaise
    english_stops = list(set(stopwords.words("english")))
    #On récupère les stopwords liés au jargon des cvs
    with open(path_static+"specific_stopwords.txt") as fichier:
        specific_stops = [stop.strip() for stop in fichier.readlines()]
    #Liste des stopwords
    stop_words = english_stops+specific_stops

    #Mauvais caractères
    Mauvais_caract = re.compile('[^0-9a-z \'#+_]')

    #Taille avant nettoyage
    #print("Nombre de termes avant nettoyage:  {} termes".format(len(" ".join(cv_type.cv))))
    # Nettoyage
    cv_type.cv = cv_type.cv.apply(lambda document: f.clean(document, stop_words, Mauvais_caract))
    #Taille après nettoyage
    #print("-----------------------------------")
    #print("Nombre de termes après nettoyage:  {} termes".format(len(" ".join(cv_type.cv))))

    X_train_vec = CountVectorizer().fit(cv_type.cv)
    X_train_trans = X_train_vec.fit_transform(cv_type.cv)
    _cv_type = X_train_trans.todense()
    df_type = pd.DataFrame(_cv_type,columns=X_train_vec.get_feature_names_out())
    don = cv_type[cv_type["_type"]=='cv_type']


    last_one=-1
    for id_ in don.ids:
        tableau=[]
        i=0
        j=1
        lst=f.recommend(id_,df_type,cv_type)
        st.write("Recommendation for : "+don.categorie[id_])
        all_cv_by_cat=[tup for tup in lst[2] if ((tup[0] < don.ids[id_]) and (tup[0]>last_one))]
        all_cv_by_cat_ = sorted(all_cv_by_cat,key=lambda x:x[1],reverse=True)
        m=f.recommend_ten(all_cv_by_cat_)
       # col1, col2, col3,col4,col5,col6,col7,col8,col9,col10 = st.columns(10)
        for el in m :
            #print(el[1])
            #df=pd.DataFrame(el[0],round(el[1],2),columns=('ID','Score de similarité')
            #st.write(str(j)+'- id : ' +str(el[0])+' ; score de similarité : '+str(round(el[1],2)))
            #)
            tableau.append([str(el[0]),round(el[1],2)])
            i=i+1
            j=j+1
        #for tab in tableau:
        
        df=pd.DataFrame(tableau,columns=("resume ID","similarity score"))
        st.dataframe(df)
        last_one =don.ids[id_]































#st.markdown("")
#st.markdown("### About this app")
#with st.expander("ℹ️ - About this app", expanded=True):
#   st.write(
 #   ""    
  #  ""
   # )
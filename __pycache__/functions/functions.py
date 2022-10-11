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
#import functions as f
import gensim
from sklearn.metrics.pairwise import cosine_similarity



def lit_cvs(chemin=""):
    """Retourne une liste de tuples ("cv au format texte", "étiquette/classe/catégorie") contenant le jeu de données"""
    _nb = 0 #On dénombre les fichiers lus
    _categories = [_cat for _cat in os.listdir(chemin) if not(_cat.startswith("desktop"))]
    st.write("#Total catégories: ", len(_categories))
    #total_cat = len(_categories)
    _cvs = list()
    #print("#Convertion des cvs de chaque catégorie...")
    for _categorie in _categories:
        _pdf_cvs= os.listdir(chemin+"/"+_categorie)
        #st.write("Cat {} : {} => {} cvs".format(_nb, _categorie, len(_pdf_cvs)), end=" ")
        #On convertit chaque pdf
        _nb_error = 0
        for _pdf_cv in _pdf_cvs:
            try:
               pdf = PyPDF2.PdfFileReader(open(chemin+"/"+_categorie+'/'+_pdf_cv, "rb"))
            except:
                #print("error")
                _nb_error+=1
                pass
            else:
                contenu=""
                for page in pdf.pages:
                    contenu+=page.extractText()
                    #print(page.extractText())
                _cvs.append((contenu, _categorie))
        #st.write(", {} error(s)".format(_nb_error))
        _nb+=1
    st.write("#Fin convertion")
    
    return _cvs

def clean (text, to_remove, regex_to_remove):
    """
        text: un document = une chaine de caractère
        to_remove: liste de mots à enlever
        regex_to_remove: une regex expression compilée
        
        retourne le texte pré-traité
    """
    # On décode tout ce qui est HTML
    text = BeautifulSoup(text, "lxml").text 
    # On uniformise la casse de notre document en minuscule
    text = text.lower() 
    #On supprimer tout ce qui est mauvais caractère
    text = regex_to_remove.sub(' ', text)
    #On supprime la liste de mots to_remove qui peut être la liste des stopwords
    text = ' '.join(word for word in text.split() if word not in to_remove)

    # On remplace chaque mots par son lemme 
    WNlemming = nltk.WordNetLemmatizer()
    text = " ".join([WNlemming.lemmatize(mot, pos="v") for mot in nltk.word_tokenize(text)])
    # On peut aussi remplacer les mots plutôt par leur racine
    #porter = nltk.PorterStemmer()
    #text = " ".join([porter.stem(mot) for mot in nltk.word_tokenize(text)])
  
    return text

def recommend(id_cv_type,df_type,cvtype):
    sim =cosine_similarity(df_type, df_type)
    cv_id=id_cv_type
    scores=list(enumerate(sim[cv_id]))
    #print(type(scores))
    id_ = [cv_id for cvs in scores]
    #sorted_scores=sorted(scores,key=lambda x:(x[1],id_),reverse=True)
    sorted_scores=sorted(scores,key=lambda x:x[0],reverse=True)
    sorted_scores=sorted_scores[1:]
    cvs=[cvtype.cv[cv_id] for cvs in sorted_scores]
    cv_typ = cvtype._type[cv_id]
    #cate=[cv_type.categorie[cv_id] for cvs in sorted_scores]
    return cv_id,cvs,sorted_scores,id_,scores

def recommend_ten(cvs):
    first_ten=[]
    count=0
    for cv in cvs:
        if count > 9:
            break
        count+=1
        first_ten.append(cv)
    return first_ten


#je récupére ici les cv_type de chaque catégories 
def lit_cvs_type(chemin=""):
    """Retourne une liste de tuples ("cv au format texte", "étiquette/classe/catégorie") contenant le jeu de données"""
    _nb = 0 #On dénombre les fichiers lus
    _categories = [_cat for _cat in os.listdir(chemin) if not(_cat.startswith("desktop"))]
    st.write("#Total catégories: ", len(_categories))
    _cvs = list()
    #st.write("#Convertion des cvs de chaque catégorie...")
    for _categorie in _categories:
        _type=''
        _pdf_cvs= os.listdir(chemin+"/"+_categorie)  
        #_pdf_cvs =  glob.glob(chemin+"/"+_categorie+'/'+"cv_type.pdf")
        #st.write("Cat {} : {} => {} cvs".format(_nb, _categorie, len(_pdf_cvs)), end=" ")
        #On convertit chaque pdf
        _nb_error = 0
        for _pdf_cv in _pdf_cvs:
            #_pdf_cv = _pdf_cv.replace("\\","/")
            #print(_pdf_cv.split('.')[0])
            try:
               #pdf = PyPDF2.PdfFileReader(open(_pdf_cv, "rb"))
               pdf = PyPDF2.PdfFileReader(open(chemin+"/"+_categorie+'/'+_pdf_cv, "rb"))
            except:
                #print("error")
                _nb_error+=1
                pass
            else:
                if _pdf_cv.split('.')[0]=='cv_type':
                    _type='cv_type'
                contenu=""
                for page in pdf.pages:
                    contenu+=page.extractText()
                    #print(page.extractText())
                _cvs.append((contenu, _categorie,_type))
        #st.write(", {} error(s)".format(_nb_error))
        _nb+=1
    st.write("#Fin convertion")
    
    return _cvs


def transform_new_CV(filename) :
    cvs_new = list()
    try:
        pdf = PyPDF2.PdfFileReader(open(filename, "rb"))
    except:
        #print("error")
        #_nb_error+=1
        pass
    else:
        i=0
        contenu=""
        for page in pdf.pages:
            contenu+=page.extractText()
        cvs_new.append(contenu)
    return cvs_new
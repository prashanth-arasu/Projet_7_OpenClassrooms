# Importation des libraries

import pandas as pd
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import streamlit.components.v1 as components
from matplotlib import image
import re

# Le modèle LGBMClassifier est importé à l'aide du Pickle

from lightgbm import LGBMClassifier

# Fonction pour la visualisation

def plot(df, col, val):
    x_cols = ['Premier', 'Deuxième', 'Troisième', 'Quatrième']
    y_bot = [df[col][0], df[col][1], df[col][2], df[col][3]]
    y_val = [df[col][1] - df[col][0], df[col][2] - df[col][1], df[col][3] - df[col][2], df[col][4] - df[col][3]]
    fig = plt.figure(figsize=(10, 5))
    graph = sns.barplot(x=x_cols, y=y_val, bottom=y_bot, alpha=0.8)
    graph.axhline(val, color='black', linestyle='-.', markeredgewidth=2.5)
    plt.title('Comparaison avec des client similaires')
    plt.ylabel(col, fontsize=12)
    plt.xlabel('Quartile', fontsize=12)
    return st.pyplot(fig)

# Fontion main

def main():
    icon = image.imread("PaD.png")
    st.title("Prêt à dépenser")
    st.image(icon)
    st.write("# Traitement des demandes du crédit")
    st.markdown("Bienvenue cher chargé du client,")
    st.markdown("- Veuillez commencer par entrer l'identifiant du client.")
    st.markdown("- Si l'identifiant est valide, vous aurez l'option de visualiser la décision sur la demande de crédit.")
    st.markdown("- Ensuite, vous pouvez choisir de visualiser l'influence des variables sur la décision et de comparer le "
                "client aux autres clients similaires.")

# Importation des dataframes

    data = pd.read_csv("app_tr.csv")
    data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    accord = pd.read_csv("app_tr_accord.csv")
    refus = pd.read_csv("app_tr_refus.csv")

# Fonction pour vérifier l'identifiant du client
    @st.cache
    def id_valid(identifiant):
        ids = data['SK_ID_CURR'].values
        if identifiant in ids:
            valid_ = "Identifiant Client Valide"
        else:
            valid_ = "Identifiant Client Non-Valide"
        return valid_

# Identifiant du client

    st.write("## Identifiant du client")
    st.markdown("> Choisir d'abord l'identifiant du client")
    identifiant = st.number_input("Entrez l'Identifiant Client:", format='%i', step=1)
    valid = id_valid(identifiant)
    st.write(id_valid(identifiant))

    if valid == "Identifiant Client Non-Valide":
        st.write("Veuillez entrer à nouveau l'identifiant du client")
    else:
        pred = st.selectbox('Voulez vous visualiser la décision sur la demande de crédit?', ('Non', 'Oui'))
        if pred == 'Oui':
            st.write('## Décision sur la demande de prêt')
            client_data = data[data.SK_ID_CURR.values == identifiant]
            client_data = client_data.drop(['SK_ID_CURR', 'TARGET'], axis=1)
            X = data.drop(['SK_ID_CURR', 'TARGET'], axis=1)
            y = data['TARGET']
            idx = data[data['SK_ID_CURR'] == identifiant].index.values

# Fonction pour la modélisation

            @st.cache
            def predict(identifiant):
                model = pickle.load(open('Pickle_LGBM_Model.pkl', 'rb'))
                model.fit(X, y)
                result_proba = model.predict_proba(client_data)
                probability1 = result_proba[0][1]
                if probability1 >= 0.14:
                    decision = 1
                else:
                    decision = 0
                return probability1, decision

# Décision sur la demande de crédit

            probability, decision = predict(identifiant)
            if decision == 1:
                st.markdown(">Probabilité du repaiement:")
                st.write(probability)
                st.markdown('>Seuil maximal:')
                st.write(0.14)
                st.markdown(">Décision sur la demande de crédit:")
                st.markdown("Client risqué: Demande de crédit refusée")
            else:
                st.markdown(">Probabilité du repaiement:")
                st.write(probability)
                st.markdown('>Seuil maximal:')
                st.write(0.14)
                st.markdown(">Décision sur la demande de crédit:")
                st.markdown("Client peu risqué: Demande de crédit accordée")

            shap_val = pickle.load(open('shap_values.pkl', 'rb'))
            exp_val = pickle.load(open('expected_shap_values.pkl', 'rb'))

# Influence des variables

            st.write('## Influence des variables')
            select = st.selectbox("Voulez vous visualiser l'influence des variables sur la décision?", ('Non', 'Oui'))
            if select == 'Oui':
                shap.initjs()
                def st_shap(plot):
                    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                    components.html(shap_html, height = None)

                st_shap(shap.force_plot(exp_val[0], shap_val[0][idx,:], X.iloc[idx,:]))
                st.write("* Les variables en bleu influencent la décision vers le refus de la demande de prêt")
                st.write("* Les variables en rouge influencent la décision vers l'accord de la demande de prêt")

# Comparaison avec des clients similaires

            st.write('## Comparaison avec des clients similaires')
            options = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'YEARS_LAST_PHONE_CHANGE', 'OWN_CAR_AGE', 'YEARS_BIRTH',
                       'REGION_POPULATION_RELATIVE']
            comparaison_option = st.selectbox('Comparaison des clients', ('--Non--', 'AMT_CREDIT', 'AMT_GOODS_PRICE',
                                                                          'YEARS_LAST_PHONE_CHANGE', 'OWN_CAR_AGE',
                                                                          'YEARS_BIRTH', 'REGION_POPULATION_RELATIVE'))
            accord = pd.read_csv("app_tr_accord.csv")
            refus = pd.read_csv("app_tr_refus.csv")

# Fonction pour la comparaison avec des clients similaires

            def comparison(decision, col):
                client_val = client_data[col].values[0]
                st.write("La valeur '", col, "'du client est: ", client_val)
                if decision == 1:
                    st.write("Comparaison avec des clients dont leur demande de prêt a été refusée")
                    st.write(refus)
                    plot(refus, col, client_val)
                else:
                    st.write("Comparaison avec des clients dont leur demande de prêt a été accordée")
                    st.write(accord)
                    plot(accord, col, client_val)

            if comparaison_option in options:
                comparison(decision, comparaison_option)

main()
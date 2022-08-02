# mise en place de FastAPI

# importation des librairies
import joblib
import pandas as pd
from fastapi import FastAPI
import streamlit as st
import uvicorn

# chargement du data_test

@st.cache(allow_output_mutation=True)
def data():
    path = "fichier_api/fichier-test1000-api.csv"
    data = pd.read_csv(path).drop("Unnamed: 0", axis=1)

    # chargement du modèle entrainé
    path_ = "fichier_api/joblib_lgbm_beta_3_Model.pkl"
    model = joblib.load(path_)

    # complétion de data_test avec score, class_bin et class_cat

    score = 100 * model.predict_proba(data.copy(deep=True).iloc[:, :-1])[:, 1]
    class_bin = model.predict(data.iloc[:, :-1])

    class_cat = []
    for i in class_bin:
        if i == 0.0:
            class_cat.append("acceptée")
        else:
            class_cat.append("refusée")

    data["score"] = score
    data["class_bin"] = class_bin
    data["class_cat"] = class_cat

    return data


app = FastAPI()  # définition de notre application


# lecture de données, (racine : "/credit")
@app.get("/credit")
async def credit(ID: int):
    """Fonction de classification d'instance en entrant que l'identifiant 'SK_ID_CURR'"""
    if ID not in data().SK_ID_CURR.values:
        return {"Veuillez saisir un identifiant client valide pour débuter l'analyse."}

    else:
        data_id = data()[data()["SK_ID_CURR"] == ID]

        return [{"score": data_id.score.values[0]}, {"Demande de crédit": data_id.class_cat.values[0]}]


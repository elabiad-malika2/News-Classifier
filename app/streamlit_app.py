import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer

# Charger modèle + embedder
model = joblib.load(r"C:\Users\elabi\OneDrive\Desktop\New Classifier\modele\news_classifier_logreg_modele.pkl")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

st.title("Classification AG News (Simple)")

text = st.text_area("Entrez un texte d'actualité :", height=150)

if st.button("Prédire"):
    if text.strip() == "":
        st.warning("Veuillez entrer un texte.")
    else:
        # Embedding
        emb = embedder.encode([text])

        # Prédiction -> nombre
        pred_num = model.predict(emb)[0]

        # --- Condition simple ---
        if pred_num == 0:
            label = "World"
        elif pred_num == 1:
            label = "Sports"
        elif pred_num == 2:
            label = "Business"
        elif pred_num == 3:
            label = "Sci/Tech"
        else:
            label = "Catégorie inconnue"

        st.success(f"**Catégorie prédite : {label}**")

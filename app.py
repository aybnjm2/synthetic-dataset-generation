import streamlit as st
import pandas as pd
import os
from src.agents.generator import AgentGenerateur
from src.utils.metrics import analyser_diversite_texte
from dotenv import load_dotenv

# Charger les variables d'environnement (pour la clé API).
load_dotenv()

# Configuration de la page Streamlit
st.set_page_config(page_title="Générateur de Données IA", layout="wide")

st.title("Générateur de Données Synthétiques")
st.markdown("""
Cette application permet d'importer un dataset réel (seeds) et de générer de nouvelles données 
synthétiques via l'API Google Gemini, tout en calculant la diversité du vocabulaire.
""")

# --- BARRE LATÉRALE (Configuration) ---
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("1. Chargez votre dataset réel (CSV)", type=["csv"])

nb_batches = st.sidebar.slider("Nombre de batches", min_value=1, max_value=10, value=3)
nb_lignes = st.sidebar.slider("Lignes à générer par batch", min_value=1, max_value=20, value=5)
nb_seeds = st.sidebar.slider("Nombre d'exemples (seeds) à envoyer", min_value=1, max_value=10, value=2)

# --- ZONE PRINCIPALE ---
if uploaded_file is not None:
    # Lecture du dataset réel
    df_reel = pd.read_csv(uploaded_file)
    
    st.subheader("Aperçu du Dataset Source")
    st.dataframe(df_reel.head())
    
    # Bouton de génération
    if st.button("Lancer la génération", type="primary"):
        
        # Vérification de la clé API
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("Erreur : La clé GOOGLE_API_KEY est introuvable dans le fichier .env")
        else:
            try:
                generateur = AgentGenerateur()
                df_finaux = []
                
                # --- NOUVEAUTÉ : GÉNÉRATION DYNAMIQUE DES PERSONAS ---
                with st.spinner("Étape 1 : Analyse des données et invention des Personas..."):
                    generateur.preparer_personas_automatiquement(df_reel)
                
                # Affichage des personas inventés par l'IA à l'utilisateur
                st.success("Analyse terminée ! Voici les contextes/personas inventés pour votre dataset :")
                st.write(generateur.personas)
                st.markdown("---")
                # -----------------------------------------------------

                # Interface de progression pour la génération
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Boucle de génération
                for i in range(nb_batches):
                    status_text.text(f"Étape 2 : Génération du batch {i+1}/{nb_batches} en cours...")
                    
                    df_batch = generateur.generer_batch(df_reel, nb_lignes=nb_lignes, nb_seeds=nb_seeds)
                    
                    if not df_batch.empty:
                        df_finaux.append(df_batch)
                    
                    # Mise à jour de la barre de progression
                    progress_bar.progress((i + 1) / nb_batches)
                
                status_text.text("Génération terminée !")
                
                # Assemblage des résultats
                if df_finaux:
                    dataset_complet = pd.concat(df_finaux, ignore_index=True)
                    
                    st.subheader("Données Synthétiques Générées")
                    st.dataframe(dataset_complet)
                    
                    # Affichage des Métriques
                    st.subheader("Métriques de Diversité (Preuve mathématique)")
                    metriques = analyser_diversite_texte(dataset_complet)
                    
                    if metriques:
                        # Afficher les métriques sous forme de colonnes visuelles
                        cols = st.columns(len(metriques))
                        for col, (cle, valeur) in zip(cols, metriques.items()):
                            col.metric(label=cle, value=valeur)
                            
                    # Bouton pour télécharger le résultat
                    csv_bytes = dataset_complet.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Télécharger le nouveau Dataset (CSV)",
                        data=csv_bytes,
                        file_name="dataset_synthetique_genere.csv",
                        mime="text/csv",
                    )
                else:
                    st.error("La génération a échoué. Aucun batch n'a renvoyé de données valides.")
            
            except Exception as e:
                st.error(f"Une erreur critique est survenue : {e}")

else:
    st.info("Veuillez uploader un fichier CSV dans la barre latérale pour commencer.")
import streamlit as st
import pandas as pd
import os
import io
from dotenv import load_dotenv

from src.agents.generator import AgentGenerateur
from src.agents.validator import AgentValidateur
from src.agents.curator import AgentCurateur
from src.utils.metrics import analyser_diversite_texte

# Charger les variables d'environnement
load_dotenv()

# Configuration de la page Streamlit
st.set_page_config(page_title="Générateur Multi-Agents (Semaine 4)", layout="wide")

st.title(" Pipeline Multi-Agents : Génération de Données Synthétiques")
st.markdown("""
Cette application orchestre 3 agents IA :
1. **L'Agent Générateur** : Invente des données basées sur des personas.
2. **L'Agent Validateur** : Vérifie le schéma et supprime les doublons (via Embeddings cosinus).
3. **L'Agent Curateur** : Formate et sauvegarde les données propres (CSV & Parquet).
""")

# --- BARRE LATÉRALE (Configuration) ---
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("1. Chargez votre dataset réel (CSV)", type=["csv"])

nb_batches = st.sidebar.slider("Nombre de batches", min_value=1, max_value=10, value=3)
nb_lignes = st.sidebar.slider("Lignes à générer par batch", min_value=1, max_value=20, value=5)
nb_seeds = st.sidebar.slider("Exemples (seeds) envoyés à l'IA", min_value=1, max_value=10, value=2)
seuil_sim = st.sidebar.slider("Seuil anti-doublon (%)", min_value=50, max_value=100, value=85) / 100.0

# --- ZONE PRINCIPALE ---
if uploaded_file is not None:
    df_reel = pd.read_csv(uploaded_file)
    
    st.subheader("Aperçu du Dataset Source")
    st.dataframe(df_reel.head())
    
    if st.button(" Lancer la Pipeline Multi-Agents", type="primary"):
        
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("Erreur : La clé GOOGLE_API_KEY est introuvable dans le fichier .env")
        else:
            try:
                # 1. Initialisation de l'équipe (Les 3 Agents)
                generateur = AgentGenerateur()
                validateur = AgentValidateur()
                curateur = AgentCurateur()
                
                df_finaux = []
                stats = {"generees": 0, "validees": 0}
                
                # ÉTAPE 1 : PERSONAS
                with st.spinner(" Étape 1 : Le Générateur analyse les données et invente des Personas..."):
                    generateur.preparer_personas_automatiquement(df_reel)
                
                st.success(" Personas inventés pour garantir la diversité :")
                st.write(generateur.personas)
                st.markdown("---")

                # Interface de progression
                progress_bar = st.progress(0)
                status_text = st.empty()
                log_container = st.container() # Pour afficher les logs des agents
                
                # BOUCLE DES AGENTS
                for i in range(nb_batches):
                    status_text.markdown(f"**Batch {i+1}/{nb_batches} en cours...**")
                    
                    with log_container:
                        # Agent Générateur
                        st.text(f" [Générateur] Création de {nb_lignes} lignes...")
                        df_batch = generateur.generer_batch(df_reel, nb_lignes=nb_lignes, nb_seeds=nb_seeds)
                        
                        if not df_batch.empty:
                            stats["generees"] += len(df_batch)
                            
                            # Agent Validateur (Format)
                            st.text(f" [Validateur] Vérification du schéma...")
                            df_valide, format_ok = validateur.valider_format(df_batch, df_reel)
                            
                            if format_ok and not df_valide.empty:
                                # Agent Validateur (Sémantique / Anti-doublons)
                                st.text(f" [Validateur] Recherche de doublons (Embeddings > {seuil_sim*100}%)...")
                                df_nettoye = validateur.valider_semantique(df_valide, seuil_similarite=seuil_sim)
                                
                                if not df_nettoye.empty:
                                    stats["validees"] += len(df_nettoye)
                                    df_finaux.append(df_nettoye)
                                    st.write(f" {len(df_nettoye)}/{len(df_batch)} lignes ont survécu à la validation.")
                                else:
                                    st.warning(f"Toutes les lignes de ce batch ont été rejetées (doublons).")
                            else:
                                st.error("Le générateur n'a pas respecté le format demandé.")
                        
                    progress_bar.progress((i + 1) / nb_batches)
                
                status_text.success("Pipeline terminée !")
                
                # ASSEMBLAGE ET CURATION
                if df_finaux:
                    dataset_complet = pd.concat(df_finaux, ignore_index=True)
                    
                    # Agent Curateur sauvegarde sur le disque
                    curateur.sauvegarder(dataset_complet, nom_fichier="dataset_multi_agents")
                    
                    st.subheader("Données Synthétiques Finales (Validées)")
                    
                    # Affichage de l'impact du Validateur
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Lignes Générées (Brut)", stats["generees"])
                    col2.metric("Lignes Validées (Propre)", stats["validees"])
                    col3.metric("Doublons/Erreurs bloqués", stats["generees"] - stats["validees"])
                    
                    st.dataframe(dataset_complet)
                    
                    st.subheader("Métriques de Diversité (Preuve mathématique)")
                    metriques = analyser_diversite_texte(dataset_complet)
                    if metriques:
                        cols = st.columns(len(metriques))
                        for col, (cle, valeur) in zip(cols, metriques.items()):
                            col.metric(label=cle, value=valeur)
                            
                    # Boutons de téléchargement (CSV et Parquet)
                    st.markdown("### Télécharger les résultats")
                    dl_col1, dl_col2 = st.columns(2)
                    
                    csv_bytes = dataset_complet.to_csv(index=False).encode('utf-8')
                    dl_col1.download_button(
                        label="⬇Télécharger en CSV (Lecture humaine)",
                        data=csv_bytes,
                        file_name="dataset_valide.csv",
                        mime="text/csv",
                    )
                    
                    # Génération du Parquet en mémoire pour Streamlit
                    parquet_buffer = io.BytesIO()
                    dataset_complet.to_parquet(parquet_buffer, engine='pyarrow', index=False)
                    dl_col2.download_button(
                        label="⬇Télécharger en Parquet (Optimisé ML/IA)",
                        data=parquet_buffer.getvalue(),
                        file_name="dataset_valide.parquet",
                        mime="application/octet-stream",
                    )
                else:
                    st.error("Échec : Le Validateur a rejeté absolument toutes les données générées.")
            
            except Exception as e:
                st.error(f"Une erreur critique est survenue : {e}")

else:
    st.info("Veuillez uploader un fichier CSV dans la barre latérale pour commencer.")
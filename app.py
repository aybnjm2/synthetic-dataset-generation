import streamlit as st
import pandas as pd
import os
import io
import time
import mlflow
import ollama

from src.agents.generator import AgentGenerateur
from src.agents.validator import AgentValidateur
from src.agents.curator import AgentCurateur
from src.utils.metrics import analyser_diversite_texte

st.set_page_config(page_title="Générateur Multi-Agents", layout="wide")


def ollama_disponible():
    """Vérifie que le serveur Ollama local répond, pour donner un message
    d'erreur clair plutôt qu'une exception brute si `ollama serve` n'est
    pas lancé."""
    try:
        ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434")).list()
        return True
    except Exception:
        return False


def afficher_telechargements(df, prefixe):
    """Boutons de téléchargement CSV + Parquet, utilisés par les 2 onglets."""
    col1, col2 = st.columns(2)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    col1.download_button(
        "⬇Télécharger en CSV", data=csv_bytes,
        file_name=f"{prefixe}.csv", mime="text/csv", key=f"dl_csv_{prefixe}",
    )
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=False)
    col2.download_button(
        "⬇Télécharger en Parquet", data=buf.getvalue(),
        file_name=f"{prefixe}.parquet", mime="application/octet-stream", key=f"dl_pq_{prefixe}",
    )


def afficher_metriques_diversite(metriques):
    """Métriques TTR/diversité en colonnes, utilisées par les 2 onglets."""
    if metriques:
        cols = st.columns(len(metriques))
        for col, (cle, valeur) in zip(cols, metriques.items()):
            col.metric(label=cle, value=valeur)


st.title(" Pipeline Multi-Agents : Génération de Données Synthétiques")
st.markdown("""
Cette application orchestre 3 agents IA :
1. **L'Agent Générateur** : Invente des données basées sur des personas.
2. **L'Agent Validateur** : Vérifie le schéma et supprime les doublons (via Embeddings cosinus, y compris vs le dataset réel).
3. **L'Agent Curateur** : Formate et sauvegarde les données propres (CSV & Parquet).
""")

# --- BARRE LATÉRALE : uniquement le fichier source, commun aux deux modes ---
st.sidebar.header("Dataset source")
uploaded_file = st.sidebar.file_uploader("Chargez votre dataset réel (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Veuillez uploader un fichier CSV dans la barre latérale pour commencer.")
    st.stop()

df_reel = pd.read_csv(uploaded_file)

st.subheader("Aperçu du Dataset Source")
st.dataframe(df_reel.head())

onglet_standard, onglet_massif = st.tabs([
    " Pipeline standard (Semaine 4)",
    " Génération massive orchestrée — LangGraph / MLflow / DVC (Semaine 5)",
])


# ======================================================================
# ONGLET 1 : PIPELINE STANDARD (séquentiel, quelques batches, exploratoire)
# ======================================================================
with onglet_standard:
    st.markdown("Pipeline séquentiel simple, pour tester rapidement sur un petit nombre de batches.")

    col_a, col_b, col_c, col_d = st.columns(4)
    nb_batches = col_a.slider("Nombre de batches", min_value=1, max_value=10, value=3, key="std_nb_batches")
    nb_lignes = col_b.slider("Lignes / batch", min_value=1, max_value=20, value=5, key="std_nb_lignes")
    nb_seeds = col_c.slider("Seeds envoyés", min_value=1, max_value=10, value=2, key="std_nb_seeds")
    seuil_sim = col_d.slider("Seuil anti-doublon (%)", min_value=50, max_value=100, value=85, key="std_seuil") / 100.0

    if st.button(" Lancer le Pipeline Standard", type="primary", key="btn_standard"):

        if not ollama_disponible():
            st.error("Erreur : Ollama semble injoignable. Lance `ollama serve` (ou vérifie OLLAMA_HOST dans .env).")
        else:
            try:
                generateur = AgentGenerateur()
                validateur = AgentValidateur()
                curateur = AgentCurateur()

                stats = {"generees": 0, "validees": 0}

                with st.spinner(" Étape 1 : Le Générateur analyse les données et invente des Personas..."):
                    generateur.preparer_personas_automatiquement(df_reel)

                st.success(" Personas inventés pour garantir la diversité :")
                st.write(generateur.personas)

                with st.spinner(" Étape 1bis : Préchargement de la mémoire anti-doublon avec le dataset réel..."):
                    validateur.initialiser_historique(df_reel)

                st.markdown("---")

                progress_bar = st.progress(0)
                status_text = st.empty()
                log_container = st.container()

                for i in range(nb_batches):
                    status_text.markdown(f"**Batch {i+1}/{nb_batches} en cours...**")

                    with log_container:
                        st.text(f" [Générateur] Création de {nb_lignes} lignes...")
                        df_batch = generateur.generer_batch(df_reel, nb_lignes=nb_lignes, nb_seeds=nb_seeds)

                        if not df_batch.empty:
                            stats["generees"] += len(df_batch)

                            st.text(" [Validateur] Vérification du schéma...")
                            df_valide, format_ok = validateur.valider_format(df_batch, df_reel)

                            if format_ok and not df_valide.empty:
                                st.text(f" [Validateur] Recherche de doublons (Embeddings > {seuil_sim*100}%)...")
                                df_nettoye = validateur.valider_semantique(df_valide, seuil_similarite=seuil_sim)

                                if not df_nettoye.empty:
                                    stats["validees"] += len(df_nettoye)
                                    curateur.recevoir_batch_valide(df_nettoye)
                                    st.write(f" {len(df_nettoye)}/{len(df_batch)} lignes ont survécu à la validation.")
                                else:
                                    st.warning("Toutes les lignes de ce batch ont été rejetées (doublons).")
                            else:
                                st.error(f"Format rejeté : {validateur.dernier_erreur_format}")

                    progress_bar.progress((i + 1) / nb_batches)
                    # Pas de pause nécessaire : Ollama tourne en local, pas de quota API.

                status_text.success("Pipeline terminée !")

                dataset_complet = curateur.obtenir_dataset_complet()

                if not dataset_complet.empty:
                    curateur.sauvegarder(dataset_complet, nom_fichier="dataset_multi_agents")

                    st.subheader("Données Synthétiques Finales (Validées)")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Lignes Générées (Brut)", stats["generees"])
                    col2.metric("Lignes Validées (Propre)", stats["validees"])
                    col3.metric("Doublons/Erreurs bloqués", stats["generees"] - stats["validees"])

                    st.dataframe(dataset_complet)

                    st.subheader("Métriques de Diversité (Preuve mathématique)")
                    afficher_metriques_diversite(analyser_diversite_texte(dataset_complet))

                    st.markdown("### Télécharger les résultats")
                    afficher_telechargements(dataset_complet, "dataset_valide")
                else:
                    st.error("Échec : Le Validateur a rejeté absolument toutes les données générées.")

            except Exception as e:
                st.error(f"Une erreur critique est survenue : {e}")


# ======================================================================
# ONGLET 2 : GÉNÉRATION MASSIVE — LangGraph (résilience) + MLflow (tracking) + DVC (versioning)
# ======================================================================
with onglet_massif:
    st.markdown("""
    Ce mode utilise le graphe d'états **LangGraph** (résilience batch par batch :
    correction automatique en cas de rejet, jusqu'à un nombre max de tentatives),
    trace chaque run dans **MLflow**, et versionne le fichier final avec **DVC**.

    ⚠️ Pour un run de plusieurs milliers de lignes, l'onglet du navigateur reste
    bloqué pendant toute la durée (pas de rafraîchissement live comme dans un
    terminal). Pour un très gros run, préférez `python -m src.main_massive` en
    ligne de commande et laissez-le tourner en arrière-plan.
    """)

    col1, col2, col3 = st.columns(3)
    nb_lignes_cible = col1.number_input("Lignes cible (total)", min_value=10, max_value=10000, value=200, step=10)
    nb_lignes_batch = col2.number_input("Lignes / batch", min_value=1, max_value=50, value=10)
    nb_seeds_massif = col3.number_input("Seeds envoyés / batch", min_value=1, max_value=10, value=3)

    col4, col5, col6 = st.columns(3)
    seuil_massif = col4.slider("Seuil anti-doublon (%)", min_value=50, max_value=100, value=85, key="massif_seuil") / 100.0
    max_tentatives = col5.slider("Tentatives max / batch", min_value=1, max_value=5, value=3)
    intervalle_checkpoint = col6.number_input("Checkpoint tous les N batches", min_value=5, max_value=100, value=20)

    tenter_dvc = st.checkbox("Versionner automatiquement avec DVC à la fin (nécessite `dvc init` déjà fait)", value=False)

    if st.button(" Lancer la Génération Massive (LangGraph)", type="primary", key="btn_massif"):

        if not ollama_disponible():
            st.error("Erreur : Ollama semble injoignable. Lance `ollama serve` (ou vérifie OLLAMA_HOST dans .env).")
        else:
            try:
                from src.orchestration.graph import construire_graph
                from src.tracking.mlflow_tracker import (
                    demarrer_run,
                    logger_metriques_batch,
                    logger_metriques_finales,
                    terminer_run,
                )

                generateur = AgentGenerateur()
                validateur = AgentValidateur()
                curateur = AgentCurateur()

                with st.spinner(" Le Générateur analyse les données et invente des Personas..."):
                    generateur.preparer_personas_automatiquement(df_reel)
                st.success("Personas générés :")
                st.write(generateur.personas)

                with st.spinner(" Préchargement de la mémoire anti-doublon avec le dataset réel..."):
                    validateur.initialiser_historique(df_reel)

                graphe = construire_graph(generateur, validateur, curateur)

                params_run = {
                    "modele": generateur.modele,
                    "temperature_generation": 0.8,
                    "temperature_personas": 0.7,
                    "nb_seeds": nb_seeds_massif,
                    "nb_lignes_par_batch": nb_lignes_batch,
                    "seuil_similarite": seuil_massif,
                    "max_tentatives_par_batch": max_tentatives,
                    "nb_lignes_cible": nb_lignes_cible,
                    "source": "streamlit",
                }
                demarrer_run(f"streamlit_{nb_lignes_cible}_lignes", params_run)
                run_actif = mlflow.active_run()

                st.markdown("---")
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                log_expander = st.expander("Logs détaillés par batch", expanded=False)

                total_lignes = 0
                total_batches_echoues = 0
                numero_batch = 0
                debut = time.time()

                while total_lignes < nb_lignes_cible:
                    numero_batch += 1
                    etat_initial = {
                        "df_source": df_reel,
                        "nb_lignes": nb_lignes_batch,
                        "nb_seeds": nb_seeds_massif,
                        "seuil_similarite": seuil_massif,
                        "max_tentatives": max_tentatives,
                        "tentative": 0,
                        "persona_actuel": None,
                        "erreur_validation": None,
                        "df_batch": None,
                        "statut": "en_cours",
                    }

                    resultat = graphe.invoke(etat_initial)

                    lignes_validees_batch = 0
                    if resultat["statut"] == "valide" and resultat["df_batch"] is not None:
                        lignes_validees_batch = len(resultat["df_batch"])
                        total_lignes += lignes_validees_batch
                        log_expander.write(f" Batch {numero_batch} : {lignes_validees_batch} lignes validées (tentative {resultat['tentative']}).")
                    else:
                        total_batches_echoues += 1
                        log_expander.write(f" Batch {numero_batch} : abandonné après {resultat['tentative']} tentatives ({resultat.get('erreur_validation')}).")

                    logger_metriques_batch(
                        numero_batch,
                        lignes_generees=nb_lignes_batch,
                        lignes_validees=lignes_validees_batch,
                        tentatives=resultat["tentative"],
                    )
                    mlflow.log_metric("total_lignes_cumulees", total_lignes, step=numero_batch)

                    status_text.markdown(f"**{total_lignes}/{nb_lignes_cible} lignes validées** — batch {numero_batch}, {total_batches_echoues} échecs")
                    progress_bar.progress(min(total_lignes / nb_lignes_cible, 1.0))

                    if numero_batch % intervalle_checkpoint == 0:
                        df_checkpoint = curateur.obtenir_dataset_complet()
                        if not df_checkpoint.empty:
                            curateur.sauvegarder(df_checkpoint, nom_fichier="checkpoint_en_cours_streamlit")

                    # Pas de pause nécessaire : Ollama tourne en local, pas de quota API.

                duree = time.time() - debut
                dataset_final = curateur.obtenir_dataset_complet()

                if not dataset_final.empty:
                    chemin_parquet = curateur.sauvegarder(dataset_final, nom_fichier="dataset_massif_streamlit")

                    st.subheader("Résultat de la génération massive")
                    col_r1, col_r2, col_r3 = st.columns(3)
                    col_r1.metric("Lignes validées", len(dataset_final))
                    col_r2.metric("Batches échoués", total_batches_echoues)
                    col_r3.metric("Durée", f"{duree:.0f}s")

                    st.dataframe(dataset_final.head(50))

                    metriques_diversite = analyser_diversite_texte(dataset_final)
                    logger_metriques_finales(metriques_diversite)
                    mlflow.log_metric("duree_totale_secondes", duree)
                    mlflow.log_metric("batches_echoues", total_batches_echoues)
                    mlflow.log_metric("total_batches", numero_batch)

                    st.subheader("Métriques de Diversité")
                    afficher_metriques_diversite(metriques_diversite)

                    if tenter_dvc:
                        from src.tracking.dvc_helper import versionner_avec_dvc
                        with st.spinner("Versionnement DVC en cours..."):
                            hash_dvc = versionner_avec_dvc(chemin_parquet)
                        if hash_dvc:
                            mlflow.set_tag("dvc_hash", hash_dvc)
                            st.success(f"Versionné avec DVC. Hash : `{hash_dvc}`")
                            st.code(
                                f"git add {chemin_parquet}.dvc\n"
                                f"git commit -m \"Nouvelle version du dataset\"\n"
                                f"dvc push",
                                language="bash",
                            )
                        else:
                            st.warning("Le versionnement DVC a échoué (vérifie que `dvc init` a été fait). Voir les logs dans le terminal.")

                    st.info(
                        f"Run MLflow enregistré : `{run_actif.info.run_id if run_actif else 'N/A'}`. "
                        "Lance `mlflow ui` dans un terminal puis ouvre http://localhost:5000 pour le consulter."
                    )

                    st.markdown("### Télécharger les résultats")
                    afficher_telechargements(dataset_final, "dataset_massif")
                else:
                    st.error("Échec : aucune ligne n'a survécu au processus de validation.")

                terminer_run()

            except Exception as e:
                st.error(f"Une erreur critique est survenue : {e}")
                if mlflow.active_run():
                    mlflow.end_run(status="FAILED")

# src/main_massive.py
import os
import time
import argparse
import pandas as pd
import mlflow

from src.agents.generator import AgentGenerateur
from src.agents.validator import AgentValidateur
from src.agents.curator import AgentCurateur
from src.utils.metrics import analyser_diversite_texte
from src.orchestration.graph import construire_graph
from src.tracking.mlflow_tracker import (
    demarrer_run,
    logger_metriques_batch,
    logger_metriques_finales,
    terminer_run,
)
from src.tracking.dvc_helper import versionner_avec_dvc


def executer_generation_massive(
    fichier_source="data/input/dataset_clients_reel.csv",
    nb_lignes_cible=5000,
    nb_lignes_par_batch=10,
    nb_seeds=3,
    seuil_similarite=0.85,
    max_tentatives=3,
    intervalle_checkpoint=20,
    nom_run="generation_massive",
):
    print(f"=== Generation massive : cible = {nb_lignes_cible} lignes ===")

    df_source = pd.read_csv(fichier_source)

    generateur = AgentGenerateur()
    validateur = AgentValidateur()
    curateur = AgentCurateur()

    generateur.preparer_personas_automatiquement(df_source)
    validateur.initialiser_historique(df_source)

    graphe = construire_graph(generateur, validateur, curateur)

    params_run = {
        "modele": generateur.modele,
        "temperature_generation": 0.8,
        "temperature_personas": 0.7,
        "nb_seeds": nb_seeds,
        "nb_lignes_par_batch": nb_lignes_par_batch,
        "seuil_similarite": seuil_similarite,
        "max_tentatives_par_batch": max_tentatives,
        "nb_lignes_cible": nb_lignes_cible,
        "nb_personas": len(generateur.personas),
        "fichier_source": fichier_source,
    }
    demarrer_run(nom_run, params_run)

    total_lignes = 0
    total_batches_echoues = 0
    numero_batch = 0
    debut = time.time()

    try:
        while total_lignes < nb_lignes_cible:
            numero_batch += 1
            etat_initial = {
                "df_source": df_source,
                "nb_lignes": nb_lignes_par_batch,
                "nb_seeds": nb_seeds,
                "seuil_similarite": seuil_similarite,
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
            else:
                total_batches_echoues += 1
                print(f"[Orchestrateur] Batch {numero_batch} abandonne apres {resultat['tentative']} tentatives.")

            logger_metriques_batch(
                numero_batch,
                lignes_generees=nb_lignes_par_batch,
                lignes_validees=lignes_validees_batch,
                tentatives=resultat["tentative"],
            )
            mlflow.log_metric("total_lignes_cumulees", total_lignes, step=numero_batch)

            print(f"[Orchestrateur] Batch {numero_batch} -> {total_lignes}/{nb_lignes_cible} lignes validees au total.")

            # Checkpoint periodique : sauvegarde intermediaire pour ne rien
            # perdre en cas de crash/coupure pendant un run de plusieurs heures.
            if numero_batch % intervalle_checkpoint == 0:
                df_checkpoint = curateur.obtenir_dataset_complet()
                if not df_checkpoint.empty:
                    curateur.sauvegarder(df_checkpoint, nom_fichier="checkpoint_en_cours")

    except KeyboardInterrupt:
        print("\n[Orchestrateur] Interruption manuelle detectee, sauvegarde de l'etat actuel...")

    duree = time.time() - debut
    dataset_final = curateur.obtenir_dataset_complet()

    if not dataset_final.empty:
        chemin_parquet = curateur.sauvegarder(dataset_final, nom_fichier="dataset_massif")

        print("\nevaluation des Metriques Globales :")
        metriques_diversite = analyser_diversite_texte(dataset_final)
        logger_metriques_finales(metriques_diversite)

        mlflow.log_metric("duree_totale_secondes", duree)
        mlflow.log_metric("batches_echoues", total_batches_echoues)
        mlflow.log_metric("total_batches", numero_batch)

        # Versioning DVC du fichier Parquet final (comme Git pour la donnee)
        hash_dvc = versionner_avec_dvc(chemin_parquet)
        if hash_dvc:
            mlflow.set_tag("dvc_hash", hash_dvc)
            mlflow.log_artifact(f"{chemin_parquet}.dvc")

        print(
            f"\n[Orchestrateur] Termine : {len(dataset_final)} lignes generees en {duree:.1f}s "
            f"({total_batches_echoues} batches abandonnes sur {numero_batch})."
        )
    else:
        print("\n[Orchestrateur] echec : aucune ligne n'a survecu au processus de validation.")

    terminer_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation massive orchestree par LangGraph et tracee par MLflow.")
    parser.add_argument("--cible", type=int, default=5000, help="Nombre de lignes a generer au total.")
    parser.add_argument("--batch", type=int, default=10, help="Nombre de lignes generees par batch.")
    parser.add_argument("--seeds", type=int, default=3, help="Nombre d'exemples reels envoyes par batch.")
    parser.add_argument("--seuil", type=float, default=0.85, help="Seuil de similarite pour le dedoublonnage.")
    parser.add_argument("--tentatives", type=int, default=3, help="Nombre max de tentatives par batch.")
    parser.add_argument("--checkpoint", type=int, default=20, help="Sauvegarde intermediaire tous les N batches.")
    args = parser.parse_args()

    executer_generation_massive(
        nb_lignes_cible=args.cible,
        nb_lignes_par_batch=args.batch,
        nb_seeds=args.seeds,
        seuil_similarite=args.seuil,
        max_tentatives=args.tentatives,
        intervalle_checkpoint=args.checkpoint,
        nom_run=f"generation_massive_{args.cible}_lignes",
    )

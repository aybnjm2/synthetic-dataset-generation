import mlflow

NOM_EXPERIENCE = "generation-donnees-synthetiques"


def demarrer_run(nom_run, params: dict):
    """Demarre un run MLflow et logge les parametres de configuration
    (modele, temperature, seuils, etc.) pour pouvoir comparer les runs
    entre eux plus tard (ex: 'V2 avec seuil 0.9 vs V1 avec seuil 0.85')."""
    mlflow.set_experiment(NOM_EXPERIENCE)
    run = mlflow.start_run(run_name=nom_run)
    mlflow.log_params(params)
    return run


def logger_metriques_batch(numero_batch, lignes_generees, lignes_validees, tentatives):
    """Logge les metriques d'un batch individuel (courbe visible dans l'UI MLflow)."""
    mlflow.log_metric("lignes_validees_batch", lignes_validees, step=numero_batch)
    mlflow.log_metric("tentatives_batch", tentatives, step=numero_batch)
    if lignes_generees:
        taux_survie = lignes_validees / lignes_generees if lignes_generees > 0 else 0
        mlflow.log_metric("taux_survie_batch", taux_survie, step=numero_batch)


def logger_metriques_finales(metriques: dict):
    """Logge les metriques de diversite calculees sur le dataset final
    (issues de analyser_diversite_texte)."""
    for cle, valeur in metriques.items():
        cle_propre = (
            cle.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        try:
            mlflow.log_metric(cle_propre, float(valeur))
        except (ValueError, TypeError):
            mlflow.log_param(cle_propre, valeur)


def terminer_run():
    mlflow.end_run()

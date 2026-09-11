import os
import pandas as pd

from src.agents.generator import AgentGenerateur
from src.agents.validator import AgentValidateur
from src.agents.curator import AgentCurateur
from src.utils.metrics import analyser_diversite_texte


def main():
    print("Demarrage du Pipeline Multi-Agents (Semaine 4)")

    # Chemins
    dossier_input = "data/input"
    fichier_reel = os.path.join(dossier_input, "dataset_clients_reel.csv")

    # 1. Verification du fichier source
    if not os.path.exists(fichier_reel):
        os.makedirs(dossier_input, exist_ok=True)
        print(f"Creation d'un dataset de demonstration dans {fichier_reel}")
        df_demo = pd.DataFrame([
            {"id": 1, "profil": "Jeune actif", "commentaire": "Super produit, je le recommande !"},
            {"id": 2, "profil": "Retraite", "commentaire": "Je n'arrive pas a comprendre le bouton."},
            {"id": 3, "profil": "Etudiant", "commentaire": "C'est un peu cher pour moi, mais qualitatif."},
        ])
        df_demo.to_csv(fichier_reel, index=False)

    df_reel = pd.read_csv(fichier_reel)

    # 2. Instanciation des agents
    generateur = AgentGenerateur()
    validateur = AgentValidateur()
    curateur = AgentCurateur()

    # Preparation
    generateur.preparer_personas_automatiquement(df_reel)

    # FIX : on precharge l'historique semantique avec le dataset reel,
    # pour que le Validateur detecte aussi les quasi-copies des vraies
    # donnees, pas seulement les doublons entre lignes generees.
    validateur.initialiser_historique(df_reel)

    lignes_totales_validees = []
    nb_batches = 3

    # 3. La Boucle de Production
    for i in range(nb_batches):
        print(f"\n---Lancement du Batch {i+1}/{nb_batches} ---")

        # etape A : Generation
        df_batch = generateur.generer_batch(df_reel, nb_lignes=5, nb_seeds=2)

        if df_batch.empty:
            continue

        # etape B : Validation Format
        df_batch_valide, format_ok = validateur.valider_format(df_batch, df_reel)
        if not format_ok or df_batch_valide.empty:
            continue

        # etape C : Validation Semantique (Anti-doublons, y compris vs dataset source)
        df_batch_nettoye = validateur.valider_semantique(df_batch_valide, seuil_similarite=0.85)

        if not df_batch_nettoye.empty:
            lignes_totales_validees.append(df_batch_nettoye)

    # 4. Finalisation
    if lignes_totales_validees:
        dataset_complet = pd.concat(lignes_totales_validees, ignore_index=True)

        print("\nevaluation des Metriques Globales :")
        analyser_diversite_texte(dataset_complet)

        # etape D : Curation (Sauvegarde professionnelle)
        curateur.sauvegarder(dataset_complet, nom_fichier="donnees_synthetiques_multi_agents")
    else:
        print("\nechec : Aucune donnee n'a survecu au processus de validation.")


if __name__ == "__main__":
    main()
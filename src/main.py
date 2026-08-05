import os
import pandas as pd

# Nouveaux imports grâce à la nouvelle structure
from agents.generator import AgentGenerateur
from utils.metrics import analyser_diversite_texte

def main():
    print("Début de la génération de données (Architecture propre)")
    
    # Nouveaux chemins vers les dossiers data/
    dossier_input = "data/input"
    dossier_output = "data/output"
    
    # Sécurité : Créer les dossiers s'ils n'existent pas
    os.makedirs(dossier_input, exist_ok=True)
    os.makedirs(dossier_output, exist_ok=True)
    
    fichier_reel = os.path.join(dossier_input, "dataset_clients_reel.csv")
    
    # Création d'un faux dataset si introuvable
    if not os.path.exists(fichier_reel):
        print(f"{fichier_reel} introuvable. Création d'un dataset de démonstration.")
        df_demo = pd.DataFrame([
            {"id": 1, "profil": "Jeune actif", "commentaire": "Super produit, je le recommande à tout le monde !"},
            {"id": 2, "profil": "Retraité", "commentaire": "Je n'arrive pas à comprendre comment fonctionne le bouton de paiement."},
            {"id": 3, "profil": "Etudiant", "commentaire": "C'est un peu cher pour moi, mais la qualité est là."}
        ])
        df_demo.to_csv(fichier_reel, index=False)
    
    df_reel = pd.read_csv(fichier_reel)
    generateur = AgentGenerateur()
    
    df_finaux = []
    nb_batches = 3
    
    for i in range(nb_batches):
        print(f"\n---Lancement du Batch {i+1}/{nb_batches} ---")
        df_batch = generateur.generer_batch(df_reel, nb_lignes=5, nb_seeds=2)
        if not df_batch.empty:
            df_finaux.append(df_batch)
            
    if df_finaux:
        dataset_complet = pd.concat(df_finaux, ignore_index=True)
        print("\nGénération terminée !")
        
        analyser_diversite_texte(dataset_complet)
        
        # Sauvegarde dans le nouveau dossier data/output/
        fichier_sortie = os.path.join(dossier_output, "donnees_synthetiques_v2_personas.csv")
        dataset_complet.to_csv(fichier_sortie, index=False)
        print(f"\nDataset complet sauvegardé dans : {fichier_sortie}")
    else:
        print("\nLa génération a échoué.")

if __name__ == "__main__":
    main()
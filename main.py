import os
import pandas as pd
from agent_generator import AgentGenerateur
from metrics import analyser_diversite_texte

def main():
    print("Début de la génération de données (Semaine 3 : Agent & Métriques)")
    
    fichier_reel = "dataset_clients_reel.csv"
    
    # Création d'un faux dataset de départ si le vrai n'existe pas (pour éviter les crashs)
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
    
    # Nous allons générer 3 "batches" avec des personas différents
    df_finaux = []
    nb_batches = 3
    
    for i in range(nb_batches):
        print(f"\n---Lancement du Batch {i+1}/{nb_batches} ---")
        # On demande 5 lignes par batch
        df_batch = generateur.generer_batch(df_reel, nb_lignes=5, nb_seeds=2)
        if not df_batch.empty:
            df_finaux.append(df_batch)
            
    # Concaténer tous les batches
    if df_finaux:
        dataset_complet = pd.concat(df_finaux, ignore_index=True)
        print("\nGénération terminée !")
        
        # Lancement des métriques de la Semaine 3
        analyser_diversite_texte(dataset_complet)
        
        # Sauvegarde
        fichier_sortie = "donnees_synthetiques_v2_personas.csv"
        dataset_complet.to_csv(fichier_sortie, index=False)
        print(f"\nDataset complet sauvegardé dans : {fichier_sortie}")
    else:
        print("\nLa génération a échoué. Aucun dataset sauvegardé.")

if __name__ == "__main__":
    main()
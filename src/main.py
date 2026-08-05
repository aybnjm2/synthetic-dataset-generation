# src/main.py
import os
import pandas as pd

from agents.generator import AgentGenerateur
from agents.validator import AgentValidateur
from agents.curator import AgentCurateur
from utils.metrics import analyser_diversite_texte

def main():
    print("Démarrage du Pipeline Multi-Agents (Semaine 4)")
    
    # Chemins
    dossier_input = "data/input"
    fichier_reel = os.path.join(dossier_input, "dataset_clients_reel.csv")
    
    # 1. Vérification du fichier source
    if not os.path.exists(fichier_reel):
        os.makedirs(dossier_input, exist_ok=True)
        print(f"Création d'un dataset de démonstration dans {fichier_reel}")
        df_demo = pd.DataFrame([
            {"id": 1, "profil": "Jeune actif", "commentaire": "Super produit, je le recommande !"},
            {"id": 2, "profil": "Retraité", "commentaire": "Je n'arrive pas à comprendre le bouton."},
            {"id": 3, "profil": "Etudiant", "commentaire": "C'est un peu cher pour moi, mais qualitatif."}
        ])
        df_demo.to_csv(fichier_reel, index=False)
    
    df_reel = pd.read_csv(fichier_reel)
    
    # 2. Instanciation des agents
    generateur = AgentGenerateur()
    validateur = AgentValidateur()
    curateur = AgentCurateur()
    
    # Préparation
    generateur.preparer_personas_automatiquement(df_reel)
    
    lignes_totales_validees = []
    nb_batches = 3
    
    # 3. La Boucle de Production
    for i in range(nb_batches):
        print(f"\n---Lancement du Batch {i+1}/{nb_batches} ---")
        
        # Étape A : Génération
        df_batch = generateur.generer_batch(df_reel, nb_lignes=5, nb_seeds=2)
        
        if df_batch.empty:
            continue
            
        # Étape B : Validation Format
        df_batch_valide, format_ok = validateur.valider_format(df_batch, df_reel)
        if not format_ok or df_batch_valide.empty:
            continue
            
        # Étape C : Validation Sémantique (Anti-doublons)
        df_batch_nettoye = validateur.valider_semantique(df_batch_valide, seuil_similarite=0.85)
        
        if not df_batch_nettoye.empty:
            lignes_totales_validees.append(df_batch_nettoye)
            
    # 4. Finalisation
    if lignes_totales_validees:
        dataset_complet = pd.concat(lignes_totales_validees, ignore_index=True)
        
        print("\nÉvaluation des Métriques Globales :")
        analyser_diversite_texte(dataset_complet)
        
        # Étape D : Curation (Sauvegarde professionnelle)
        curateur.sauvegarder(dataset_complet, nom_fichier="donnees_synthetiques_multi_agents")
    else:
        print("\nÉchec : Aucune donnée n'a survécu au processus de validation.")

if __name__ == "__main__":
    main()
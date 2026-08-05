# src/agents/curator.py
import pandas as pd
import os

class AgentCurateur:
    def __init__(self, dossier_sortie="data/output"):
        self.dossier_sortie = dossier_sortie
        os.makedirs(self.dossier_sortie, exist_ok=True)

    def sauvegarder(self, df_final, nom_fichier="dataset_final"):
        """Sauvegarde les données propres en format Parquet et CSV (pour visualisation)."""
        print("\n[Curateur] Formatage et sauvegarde des données...")
        
        chemin_parquet = os.path.join(self.dossier_sortie, f"{nom_fichier}.parquet")
        chemin_csv = os.path.join(self.dossier_sortie, f"{nom_fichier}.csv")
        
        # Sauvegarde Parquet (Le format idéal ML)
        df_final.to_parquet(chemin_parquet, engine="pyarrow", index=False)
        
        # Sauvegarde CSV (Pour que l'humain puisse relire facilement)
        df_final.to_csv(chemin_csv, index=False)
        
        print(f"✨ [Curateur] Succès ! Données enregistrées dans :")
        print(f"   - {chemin_parquet} (Optimisé IA)")
        print(f"   - {chemin_csv} (Pour lecture humaine)")
# src/agents/curator.py
import pandas as pd
import os


class AgentCurateur:
    def __init__(self, dossier_sortie="data/output"):
        self.dossier_sortie = dossier_sortie
        os.makedirs(self.dossier_sortie, exist_ok=True)

        # buffer en memoire des batches valides par le graphe LangGraph
        # (le Curateur est appele a chaque fois que le Validateur dit "OK").
        # La sauvegarde disque reelle se fait via checkpoints periodiques
        # plutot qu'a chaque batch pour eviter de reecrire tout le fichier
        # des milliers de fois lors d'une generation massive.
        self.buffer_batches = []

    def recevoir_batch_valide(self, df_batch):
        """Appelé par le nœud 'curer' du graphe quand un batch a été validé
        (format + sémantique OK). Accumule le batch en mémoire."""
        if df_batch is not None and not df_batch.empty:
            self.buffer_batches.append(df_batch)
        return len(df_batch) if df_batch is not None else 0

    def obtenir_dataset_complet(self):
        """Retourne tous les batches validés accumulés jusqu'ici, concaténés."""
        if not self.buffer_batches:
            return pd.DataFrame()
        return pd.concat(self.buffer_batches, ignore_index=True)

    def sauvegarder(self, df_final=None, nom_fichier="dataset_final"):
        """Sauvegarde les données propres en format Parquet et CSV.

        Si df_final n'est pas fourni, sauvegarde le buffer accumulé via
        recevoir_batch_valide (pratique pour les checkpoints périodiques
        et la sauvegarde finale d'un run massif).

        Retourne le chemin du fichier Parquet généré.
        """
        if df_final is None:
            df_final = self.obtenir_dataset_complet()

        if df_final is None or df_final.empty:
            print("[Curateur] Rien à sauvegarder (dataset vide).")
            return None

        print("\n[Curateur] Formatage et sauvegarde des données...")

        chemin_parquet = os.path.join(self.dossier_sortie, f"{nom_fichier}.parquet")
        chemin_csv = os.path.join(self.dossier_sortie, f"{nom_fichier}.csv")

        df_final.to_parquet(chemin_parquet, engine="pyarrow", index=False)
        df_final.to_csv(chemin_csv, index=False)

        print(f" [Curateur] Succès ! Données enregistrées dans :")
        print(f"   - {chemin_parquet} (Optimisé IA)")
        print(f"   - {chemin_csv} (Pour lecture humaine)")

        return chemin_parquet

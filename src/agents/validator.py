# src/agents/validator.py
import os
import pandas as pd
import pandera as pa
import numpy as np
import ollama
from sklearn.metrics.pairwise import cosine_similarity


class AgentValidateur:
    def __init__(self):
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = ollama.Client(host=self.host)

        # nomic-embed-text : leger (274 Mo) rapide largement suffisant
        # pour de la detection de similarite/doublons.
        self.embedding_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

        # memoire de l'agent  on stocke les vecteurs deja valides (batches
        # precedents + dataset source) pour eviter les repetitions.
        self.historique_embeddings = []

        # tracabilite pour l'orchestrateur (LangGraph) : dernier message
        # d'erreur de format rencontre a renvoyer au generateur pour
        # qu'il puisse corriger sa prochaine tentative.
        self.dernier_erreur_format = None

    def initialiser_historique(self, df_source):
        """Embed les lignes du dataset réel pour que le futur dédoublonnage
        sémantique détecte aussi les quasi-copies des vraies données."""
        colonnes_texte = df_source.select_dtypes(include=["object"]).columns
        if len(colonnes_texte) == 0:
            print("[Validateur] Pas de colonnes texte dans la source, pas de préchargement sémantique.")
            return

        textes_source = []
        for _, row in df_source.iterrows():
            textes_source.append(" ".join(str(row[col]) for col in colonnes_texte))

        try:
            print(f"[Validateur] Préchargement de {len(textes_source)} lignes sources dans l'historique...")
            response = self.client.embed(model=self.embedding_model, input=textes_source)
            for emb in response["embeddings"]:
                self.historique_embeddings.append(np.array(emb))
            print("[Validateur] Historique initialisé avec le dataset source.")
        except Exception as e:
            print(f"[Validateur] Erreur lors du préchargement de l'historique source : {e}")

    def valider_format(self, df_genere, df_source):
        """Vérifie que les données générées ont la même structure, sans bloquer les nouvelles valeurs (min/max)."""
        print("[Validateur] Vérification du format (Schéma)...")
        try:
            schema_attendu = pa.infer_schema(df_source)

            for nom_colonne, colonne in schema_attendu.columns.items():
                colonne.checks = []

            # mode strict : rejette toute colonne en trop que le modele
            # aurait pu halluciner (non declaree dans le schéma source).
            schema_attendu.strict = True

            # FIX : l'index infere depuis l'echantillon source contient
            # souvent une contrainte parasite du type "index <= N" (borne
            # du nombre de lignes de l'echantillon), qui n'a rien a voir
            # avec les donnees elles-memes. sans ce fix tout batch genere
            # avec plus de lignes que le dataset source se fait rejeter
            # systematiquement sur les lignes en trop.
            schema_attendu.index = None

            df_valide = schema_attendu.validate(df_genere)
            self.dernier_erreur_format = None
            return df_valide, True

        except pa.errors.SchemaErrors as e:
            message = f"Le format ne correspond pas au schéma attendu (plusieurs erreurs) : {e}"
            print(f"[Validateur] Rejeté ! {message}")
            self.dernier_erreur_format = message
            return pd.DataFrame(), False

        except pa.errors.SchemaError as e:
            message = f"Colonne manquante, en trop, ou mal typée par rapport au dataset source : {e}"
            print(f"[Validateur] Rejeté ! {message}")
            self.dernier_erreur_format = message
            return pd.DataFrame(), False

        except Exception as e:
            message = f"Erreur inattendue lors de la validation du format : {e}"
            print(f"[Validateur] Rejeté ! {message}")
            self.dernier_erreur_format = message
            return pd.DataFrame(), False

    def valider_semantique(self, df_genere, seuil_similarite=0.85):
        """Embed le batch en UNE SEULE requête groupée (API /api/embed d'Ollama,
        supporte les listes en entrée) et rejette les quasi-doublons."""
        print("[Validateur] Analyse sémantique (Embeddings groupés)...")

        colonnes_texte = df_genere.select_dtypes(include=["object"]).columns

        if len(colonnes_texte) == 0:
            print("[Validateur] Aucune colonne texte : dédoublonnage exact appliqué en repli.")
            avant = len(df_genere)
            df_dedupe = df_genere.drop_duplicates().reset_index(drop=True)
            print(f"[Validateur] {len(df_dedupe)}/{avant} lignes uniques conservées.")
            return df_dedupe

        textes_a_embed = []
        for index, row in df_genere.iterrows():
            texte_concat = " ".join(str(row[col]) for col in colonnes_texte)
            textes_a_embed.append(texte_concat)

        lignes_valides = []

        try:
            response = self.client.embed(model=self.embedding_model, input=textes_a_embed)
            embeddings = response["embeddings"]

            for i, row in enumerate(df_genere.itertuples(index=False)):
                vecteur_actuel = np.array(embeddings[i]).reshape(1, -1)
                est_un_doublon = False

                if len(self.historique_embeddings) > 0:
                    historique_matrice = np.vstack(self.historique_embeddings)
                    similarites = cosine_similarity(vecteur_actuel, historique_matrice)[0]

                    if any(score > seuil_similarite for score in similarites):
                        est_un_doublon = True

                if not est_un_doublon:
                    self.historique_embeddings.append(vecteur_actuel[0])
                    lignes_valides.append(row._asdict())
                else:
                    print("[Validateur] Ligne rejetée (Doublon)")

        except Exception as e:
            print(f"[Validateur] Erreur API Embedding : {e}")
            return df_genere

        df_final = pd.DataFrame(lignes_valides)
        if not df_final.empty:
            df_final = df_final.astype(df_genere.dtypes.to_dict())

        print(f"[Validateur] {len(df_final)}/{len(df_genere)} lignes ont survécu à la validation.")
        return df_final

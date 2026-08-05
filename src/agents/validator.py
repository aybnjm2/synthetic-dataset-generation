# src/agents/validator.py
import os
import pandas as pd
import pandera as pa
import numpy as np
from google import genai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

class AgentValidateur:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("La clé GOOGLE_API_KEY est introuvable.")
        self.client = genai.Client(api_key=api_key)
        
        # Modèle d'embedding de Google (spécialisé pour transformer du texte en vecteurs)
        self.embedding_model = "text-embedding-004"
        
        # Mémoire de l'agent : on stocke les vecteurs déjà validés pour éviter 
        # que le batch 3 ne répète ce qu'a dit le batch 1.
        self.historique_embeddings = []

    def valider_format(self, df_genere, df_source):
        """Vérifie que les données générées ont EXACTEMENT la même structure que les vraies données."""
        print("[Validateur] Vérification du format (Schéma)...")
        try:
            # On infère le schéma idéal depuis le dataset source
            schema_attendu = pa.infer_schema(df_source)
            # On vérifie si le dataframe généré correspond
            df_valide = schema_attendu.validate(df_genere)
            return df_valide, True
        except pa.errors.SchemaError as e:
            print(f"[Validateur] Rejeté ! Erreur de schéma : {e}")
            return pd.DataFrame(), False

    def valider_semantique(self, df_genere, seuil_similarite=0.85):
        """Transforme les textes en vecteurs et rejette les doublons sémantiques."""
        print("[Validateur] Analyse sémantique (Embeddings)...")
        
        # On concatène toutes les colonnes textuelles de la ligne pour faire un "résumé" de la ligne
        colonnes_texte = df_genere.select_dtypes(include=['object']).columns
        if len(colonnes_texte) == 0:
            return df_genere  # S'il n'y a pas de texte, on ne peut pas faire de NLP

        lignes_valides = []
        
        for index, row in df_genere.iterrows():
            texte_concat = " ".join(str(row[col]) for col in colonnes_texte)
            
            try:
                # Appel à Gemini pour avoir l'embedding
                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=texte_concat
                )
                vecteur_actuel = np.array(response.embeddings[0].values).reshape(1, -1)
                
                est_un_doublon = False
                
                # Comparaison avec l'historique
                if len(self.historique_embeddings) > 0:
                    historique_matrice = np.vstack(self.historique_embeddings)
                    similarites = cosine_similarity(vecteur_actuel, historique_matrice)[0]
                    
                    # Si la ressemblance avec un texte précédent dépasse le seuil (ex: 85%)
                    if any(score > seuil_similarite for score in similarites):
                        est_un_doublon = True
                        
                if not est_un_doublon:
                    self.historique_embeddings.append(vecteur_actuel[0])
                    lignes_valides.append(row)
                else:
                    print(f"[Validateur] Ligne rejetée (Doublon sémantique détecté)")
                    
            except Exception as e:
                print(f"[Validateur] Erreur API Embedding : {e}")
                # En cas d'erreur de l'API, on est indulgent et on garde la ligne
                lignes_valides.append(row)

        df_final = pd.DataFrame(lignes_valides)
        if not df_final.empty:
             # On s'assure de conserver les mêmes types (les itérations pandas peuvent parfois altérer les types)
             df_final = df_final.astype(df_genere.dtypes.to_dict())
             
        print(f"[Validateur] {len(df_final)}/{len(df_genere)} lignes ont survécu à la validation.")
        return df_final
import os
import json
import random
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

class AgentGenerateur:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("La clé GOOGLE_API_KEY est introuvable.")
        self.client = genai.Client(api_key=api_key)
        
        # Liste de personas pour forcer la diversité des données générées
        self.personas = [
            "Un client très mécontent et agressif",
            "Un étudiant avec un budget très limité",
            "Un professionnel pressé qui utilise un langage formel",
            "Un utilisateur novice, confus et qui pose beaucoup de questions",
            "Un client fidèle et très enthousiaste",
            "Une personne âgée qui n'est pas à l'aise avec la technologie"
        ]

    def generer_batch(self, df_source, nb_lignes=5, nb_seeds=3):
        # 1. Sélection d'un persona au hasard
        persona_actuel = random.choice(self.personas)
        print(f"Persona sélectionné pour ce batch : '{persona_actuel}'")
        
        # 2. Sélection des seeds (graines)
        seeds = df_source.sample(n=nb_seeds).to_dict(orient="records")
        seeds_str = json.dumps(seeds, indent=2, ensure_ascii=False)

        # 3. Prompt Système mis à jour avec le Persona
        system_prompt = f"""Tu es un Expert en Data Science spécialisé en Data Augmentation.
Ton objectif est de générer de nouvelles lignes de données synthétiques pour entraîner un modèle de Machine Learning.

Règles strictes :
1. Respecte EXACTEMENT le même schéma de colonnes et types que les exemples fournis.
2. CONTRAINTE MAJEURE : Tu dois inventer de nouvelles données en adoptant ce contexte/persona : "{persona_actuel}".
   Adapte les valeurs textuelles (commentaires, questions, notes, etc.) pour refléter ce comportement.
3. Ne copie pas les exemples, invente des valeurs inédites.
4. Ton JSON doit obligatoirement avoir cette structure : {{"donnees": [ {{ligne1}}, {{ligne2}} ]}}."""

        # 4. Prompt Utilisateur
        user_prompt = f"Exemples réels :\n{seeds_str}\n\nGénère {nb_lignes} nouvelles lignes synthétiques inédites."

        # 5. Configuration et appel API
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            temperature=0.8 # Température un peu plus haute pour plus de créativité
        )

        response = self.client.models.generate_content(
            model="gemini-3.5-flash", # Vous utilisiez 3.5, mais 1.5-flash est le nom correct actuel du modèle rapide chez Google
            contents=user_prompt,
            config=config
        )
        
        try:
            # NETTOYAGE AJOUTÉ : Enlever les éventuelles balises Markdown ```json et ```
            reponse_texte = response.text.strip()
            if reponse_texte.startswith("```json"):
                reponse_texte = reponse_texte[7:]
            if reponse_texte.endswith("```"):
                reponse_texte = reponse_texte[:-3]

            donnees_json = json.loads(response.text)
            if "donnees" not in donnees_json:
                raise KeyError("La clé 'donnees' est absente du JSON.")
            return pd.DataFrame(donnees_json["donnees"])
        except Exception as e:
            print("Erreur de génération avec ce batch.")
            return pd.DataFrame() # Retourne un DF vide en cas d'erreur
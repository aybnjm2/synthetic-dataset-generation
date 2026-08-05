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
        
        # Les personas sont maintenant vides au départ.
        # L'Agent va les inventer lui-même selon le dataset.
        self.personas = []
        self.modele = "gemini-1.5-flash" # Le bon modèle, rapide et pas cher

    def preparer_personas_automatiquement(self, df_source):
        """Demande à l'IA d'analyser le dataset et d'inventer des personas adaptés."""
        print("🧠 Analyse du dataset en cours pour générer des personas sur mesure...")
        
        # Prendre 3 lignes au hasard pour que l'IA comprenne de quoi parle le dataset
        echantillon = df_source.sample(min(3, len(df_source))).to_dict(orient="records")
        echantillon_str = json.dumps(echantillon, indent=2, ensure_ascii=False)

        prompt = f"""Voici un échantillon d'un jeu de données inconnu :
{echantillon_str}

1. Identifie de quoi parle ce jeu de données (quel est le domaine métier ?).
2. Génère 5 "personas", "situations" ou "contextes" très différents qui pourraient générer ce type de données. 
   (Par exemple, si ce sont des données de banque, un persona pourrait être "Un jeune qui ouvre son premier compte").
3. Tu dois OBLIGATOIREMENT répondre sous la forme d'un JSON strict avec cette structure exacte :
{{"personas": ["persona 1", "persona 2", "persona 3", "persona 4", "persona 5"]}}"""

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.7
        )

        try:
            response = self.client.models.generate_content(
                model=self.modele,
                contents=prompt,
                config=config
            )
            
            # Nettoyage de sécurité (au cas où il y a des balises Markdown ```json)
            texte_propre = response.text.strip()
            if texte_propre.startswith("```json"):
                texte_propre = texte_propre[7:]
            if texte_propre.endswith("```"):
                texte_propre = texte_propre[:-3]
                
            resultat_json = json.loads(texte_propre)
            self.personas = resultat_json.get("personas", [])
            
            print("\n🎯 Personas générés automatiquement pour ce dataset :")
            for p in self.personas:
                print(f"  - {p}")
                
        except Exception as e:
            print(f"⚠️ Erreur lors de la création des personas : {e}")
            # Personas de secours (fallback) si l'IA plante
            self.personas = ["Cas standard", "Cas atypique", "Cas avec des valeurs extrêmes"]

    def generer_batch(self, df_source, nb_lignes=5, nb_seeds=3):
        # Sécurité : si on a oublié de préparer les personas, on utilise une valeur par défaut
        if not self.personas:
            self.personas = ["Cas standard"]
            
        persona_actuel = random.choice(self.personas)
        print(f"\n🎭 Persona sélectionné pour ce batch : '{persona_actuel}'")
        
        seeds = df_source.sample(min(nb_seeds, len(df_source))).to_dict(orient="records")
        seeds_str = json.dumps(seeds, indent=2, ensure_ascii=False)

        system_prompt = f"""Tu es un Expert en Data Science spécialisé en Data Augmentation.
Ton objectif est de générer de nouvelles lignes de données synthétiques pour entraîner un modèle de Machine Learning.

Règles strictes :
1. Respecte EXACTEMENT le même schéma de colonnes et types que les exemples fournis.
2. CONTRAINTE MAJEURE : Tu dois inventer de nouvelles données en adoptant ce contexte/persona : "{persona_actuel}".
3. Ne copie pas les exemples, invente des valeurs inédites cohérentes avec le persona.
4. Ton JSON doit obligatoirement avoir cette structure : {{"donnees": [ {{ligne1}}, {{ligne2}} ]}}."""

        user_prompt = f"Exemples réels :\n{seeds_str}\n\nGénère {nb_lignes} nouvelles lignes synthétiques inédites."

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            temperature=0.8
        )

        try:
            response = self.client.models.generate_content(
                model=self.modele,
                contents=user_prompt,
                config=config
            )
            
            texte_propre = response.text.strip()
            if texte_propre.startswith("```json"):
                texte_propre = texte_propre[7:]
            if texte_propre.endswith("```"):
                texte_propre = texte_propre[:-3]
                
            donnees_json = json.loads(texte_propre)
            return pd.DataFrame(donnees_json["donnees"])
            
        except Exception as e:
            print(f"❌ Erreur de génération avec ce batch : {e}")
            return pd.DataFrame()
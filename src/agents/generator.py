# src/agents/generator.py
import os
import json
import random
import pandas as pd
import ollama


class AgentGenerateur:
    def __init__(self):
        # Ollama tourne en local aucune cle API necessaire
        # OLLAMA_HOST permet de pointer vers un autre serveur (ex: machine
        # distante avec GPU) si besoin sinon localhost par defaut
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = ollama.Client(host=self.host)

        # qwen2.5:7b-instruct : bon compromis qualit/vitesse sur 8-12 Go VRAM
        # fiable en français et sur la génération JSON structurée.
        self.modele = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

        self.personas = []
        self.dernier_persona_utilise = None

    def preparer_personas_automatiquement(self, df_source):
        """Demande au modèle d'analyser le dataset et d'inventer des personas adaptés."""
        print("Analyse du dataset en cours pour générer des personas sur mesure...")

        echantillon = df_source.sample(min(3, len(df_source))).to_dict(orient="records")
        echantillon_str = json.dumps(echantillon, indent=2, ensure_ascii=False)

        prompt = f"""Voici un échantillon d'un jeu de données inconnu :
{echantillon_str}

1. Identifie de quoi parle ce jeu de données (quel est le domaine métier ?).
2. Génère 5 "personas", "situations" ou "contextes" très différents qui pourraient générer ce type de données. 
   (Par exemple, si ce sont des données de banque, un persona pourrait être "Un jeune qui ouvre son premier compte").
3. Tu dois OBLIGATOIREMENT répondre sous la forme d'un JSON strict avec cette structure exacte :
{{"personas": ["persona 1", "persona 2", "persona 3", "persona 4", "persona 5"]}}"""

        try:
            response = self.client.generate(
                model=self.modele,
                prompt=prompt,
                format="json",  # force une sortie JSON valide (syntaxe pas le schema exact)
                options={"temperature": 0.7},
            )
            resultat_json = json.loads(response["response"])
            self.personas = resultat_json.get("personas", [])

            print("\nPersonas générés automatiquement pour ce dataset :")
            for p in self.personas:
                print(f"  - {p}")

        except Exception as e:
            print(f"Erreur lors de la création des personas : {e}")
            self.personas = ["Cas standard", "Cas atypique", "Cas avec des valeurs extrêmes"]

    def generer_batch(self, df_source, nb_lignes=5, nb_seeds=3, persona=None, erreur_precedente=None):
        """Génère un batch de données synthétiques.

        persona : si fourni, force ce persona (utile pour les tentatives de
        correction sur un même batch, afin de garder le même contexte).
        erreur_precedente : message décrivant pourquoi la tentative
        précédente a été rejetée par le Validateur ; injecté dans le prompt
        pour que le modèle corrige explicitement le problème. Avec un modèle
        local, ce feedback est particulièrement utile : contrairement à
        Gemini, un modèle 7-8B respecte moins scrupuleusement un schéma JSON
        complexe du premier coup.
        """
        if not self.personas:
            self.personas = ["Cas standard"]

        persona_actuel = persona or random.choice(self.personas)
        self.dernier_persona_utilise = persona_actuel
        print(f"\nPersona sélectionné pour ce batch : '{persona_actuel}'")

        seeds = df_source.sample(min(nb_seeds, len(df_source))).to_dict(orient="records")
        seeds_str = json.dumps(seeds, indent=2, ensure_ascii=False)

        system_prompt = f"""Tu es un Expert en Data Science spécialisé en Data Augmentation.
Ton objectif est de générer de nouvelles lignes de données synthétiques pour entraîner un modèle de Machine Learning.

Règles strictes :
1. Respecte EXACTEMENT le même schéma de colonnes et types que les exemples fournis.
2. CONTRAINTE MAJEURE : Tu dois inventer de nouvelles données en adoptant ce contexte/persona : "{persona_actuel}".
3. Ne copie pas les exemples. INVENTE des valeurs inédites.
4. VARIÉTÉ EXTRÊME : Assure-toi de varier au maximum les combinaisons (change les genres, les notes, les situations financières à chaque ligne). Ne génère jamais deux étudiants qui se ressemblent.
5. Ton JSON doit obligatoirement avoir cette structure : {{"donnees": [ {{ligne1}}, {{ligne2}} ]}}."""

        correction = ""
        if erreur_precedente:
            correction = f"""

ATTENTION - CORRECTION OBLIGATOIRE :
Ta tentative précédente pour ce batch a été REJETÉE par le système de validation pour la raison suivante :
"{erreur_precedente}"
Tu dois impérativement corriger ce problème précis dans cette nouvelle génération, tout en respectant les règles ci-dessus."""

        user_prompt = (
            f"Exemples réels :\n{seeds_str}\n\n"
            f"Génère {nb_lignes} nouvelles lignes synthétiques inédites.{correction}"
        )

        try:
            response = self.client.generate(
                model=self.modele,
                system=system_prompt,
                prompt=user_prompt,
                format="json",
                options={"temperature": 0.8},
            )
            donnees_json = json.loads(response["response"])
            return pd.DataFrame(donnees_json["donnees"])

        except Exception as e:
            print(f"Erreur lors de la génération du batch : {e}")
            return pd.DataFrame()

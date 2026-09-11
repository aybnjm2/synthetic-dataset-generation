import pandas as pd
import re

def analyser_diversite_texte(df):
    """
    Analyse les colonnes textuelles d'un DataFrame pour calculer 
    la richesse du vocabulaire (Type-Token Ratio) et la longueur moyenne.
    """
    print("\nCalcul des metriques de diversite...")
    
    # Identifier les colonnes qui contiennent du texte (chaînes de caracteres)
    colonnes_textes = df.select_dtypes(include=['object']).columns
    
    if len(colonnes_textes) == 0:
        print("Aucune colonne de texte trouvee pour calculer les metriques.")
        return {}

    tous_les_textes = ""
    for col in colonnes_textes:
        tous_les_textes += " " + " ".join(df[col].dropna().astype(str).tolist())

    # Nettoyage basique (minuscules, on garde que les mots)
    mots = re.findall(r'\b\w+\b', tous_les_textes.lower())
    
    nb_mots_total = len(mots)
    nb_mots_uniques = len(set(mots))
    
    # Calcul de la longueur moyenne par ligne (en mots)
    longueur_moyenne_ligne = nb_mots_total / len(df) if len(df) > 0 else 0
    
    # TTR (Type-Token Ratio) : Richesse du vocabulaire
    # Un score proche de 1 signifie que chaque mot utilise est unique.
    # Un score proche de 0 signifie que l'IA repete toujours les mêmes mots.
    ttr = (nb_mots_uniques / nb_mots_total) if nb_mots_total > 0 else 0

    metriques = {
        "Total Lignes": len(df),
        "Total Mots": nb_mots_total,
        "Mots Uniques": nb_mots_uniques,
        "Mots / Ligne (Moyenne)": round(longueur_moyenne_ligne, 2),
        "Richesse Vocabulaire (TTR)": round(ttr, 4)
    }

    for cle, valeur in metriques.items():
        print(f"   - {cle} : {valeur}")
        
    return metriques
from typing import TypedDict, Optional
import pandas as pd


class PipelineState(TypedDict):
    df_source: pd.DataFrame          # dataset reel (jamais modifie)
    nb_lignes: int                   # nb de lignes a generer pour ce batch
    nb_seeds: int                    # nb d'exemples reels envoyes au LLM
    seuil_similarite: float          # seuil de dedoublonnage semantique

    max_tentatives: int              # nb max de tentatives avant abandon
    tentative: int                   # tentative courante

    persona_actuel: Optional[str]    # persona utilise (fixe entre les retries)
    erreur_validation: Optional[str]  # message d'erreur a corriger si rejet

    df_batch: Optional[pd.DataFrame]  # dernier batch produit / filtre

    # "en_cours" | "valide" | "invalide_semantique" | "echec"
    statut: str

# Pipeline Multi-Agents — Semaine 5 : Orchestration & Passage à l'échelle

## 1. Installation

```bash
pip install -r requirements.txt
```

## 2. Mise en place de DVC (remote local)

Ces commandes ne s'exécutent qu'**une seule fois** à la racine du projet.

```bash
# Si le projet n'est pas encore un repo Git
git init

# Initialiser DVC
dvc init
git add .dvc .dvcignore
git commit -m "Initialisation de DVC"

# Configurer un remote LOCAL (un simple dossier sur ta machine, hors du repo Git)
dvc remote add -d stockage_local ../dvc-storage
git add .dvc/config
git commit -m "Configuration du remote DVC local"
```

À chaque génération massive, `src/main_massive.py` exécute automatiquement
`dvc add data/output/dataset_massif.parquet`. Il te reste à committer et
pousser manuellement (le script ne le fait pas automatiquement pour te
laisser le contrôle) :

```bash
git add data/output/dataset_massif.parquet.dvc
git commit -m "Nouvelle version du dataset (voir run MLflow associé)"
dvc push
```

## 3. Suivi des expériences avec MLflow

Les runs sont stockés localement dans `./mlruns` (aucune config nécessaire).
Pour visualiser le dashboard :

```bash
mlflow ui
```

Puis ouvre http://localhost:5000 — tu y trouveras l'expérience
`generation-donnees-synthetiques` avec, pour chaque run : les paramètres
(modèle, température, seuils, nb_seeds...), les métriques par batch et
finales (taux de survie, diversité TTR...), et un tag `dvc_hash` qui relie
le run à la version exacte des données produites (traçabilité complète
"quel prompt/config a produit quelle donnée").

## 4. Lancer une génération massive

```bash
python -m src.main_massive --cible 5000 --batch 10 --seeds 3 --seuil 0.85 --tentatives 3
```

Options :
- `--cible` : nombre total de lignes validées à atteindre (5000, 10000...)
- `--batch` : lignes générées par appel au LLM (10-20 recommandé, au-delà l'API d'embedding peut refuser le batch)
- `--seeds` : exemples réels envoyés au LLM par batch
- `--seuil` : seuil de similarité cosinus pour le dédoublonnage sémantique
- `--tentatives` : nombre max d'essais de correction par batch avant abandon
- `--checkpoint` : fréquence de sauvegarde intermédiaire (en nb de batches)

Le script sauvegarde périodiquement dans `data/output/checkpoint_en_cours.*`
(résilience en cas de coupure), puis à la fin dans
`data/output/dataset_massif.parquet` / `.csv`, versionné avec DVC et tracé
dans MLflow.

## 5. Architecture du graphe LangGraph

```
generer → valider_format ─(OK)→ valider_semantique ─(OK)→ curer → FIN
              │  (KO, retries restants)   │  (KO, retries restants)
              └──────────→ generer ←──────┘
              │  (KO, plus de retries)    │  (KO, plus de retries)
              └──────────→ echec → FIN ←──┘
```

Le graphe traite **un batch à la fois** avec sa propre logique de
correction (le message d'erreur du Validateur est réinjecté dans le prompt
du Générateur). L'orchestrateur (`src/main_massive.py`) appelle ce graphe
en boucle jusqu'à atteindre le nombre de lignes cible — c'est lui qui gère
le passage à l'échelle, pendant que le graphe garantit la résilience batch
par batch.

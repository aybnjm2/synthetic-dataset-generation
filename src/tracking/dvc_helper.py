import subprocess
import yaml


def versionner_avec_dvc(chemin_fichier):
    print(f"[DVC] Versionnement de {chemin_fichier}...")
    resultat = subprocess.run(
        ["dvc", "add", chemin_fichier],
        capture_output=True,
        text=True,
    )

    if resultat.returncode != 0:
        print(f"[DVC] Erreur lors de 'dvc add' : {resultat.stderr}")
        return None

    chemin_dvc = f"{chemin_fichier}.dvc"
    try:
        with open(chemin_dvc, "r") as f:
            meta = yaml.safe_load(f)
        md5_hash = meta["outs"][0]["md5"]
        print(f"[DVC] Fichier versionne avec succès. Hash : {md5_hash}")
        print(f"[DVC] Pense a pousser les donnees : dvc push")
        print(f"[DVC] Et a committer le pointeur Git : git add {chemin_dvc} && git commit -m '...'")
        return md5_hash
    except Exception as e:
        print(f"[DVC] Impossible de lire le hash depuis {chemin_dvc} : {e}")
        return None

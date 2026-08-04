FROM python:3.10-slim

# Installer uv dans le conteneur (ultra rapide)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copier les requirements et utiliser uv pip install
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copier le reste du code
COPY . .

# Lancer le chef d'orchestre
CMD ["python", "main.py"]
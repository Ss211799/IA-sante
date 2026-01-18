from pathlib import Path

# src/clinical_tool/config.py -> parent -> src -> parent -> racine
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Définition des chemins clés
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Optionnel : Création automatique des dossiers s'ils n'existent pas
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
"""Repository-relative interface configuration; no model imports."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = Path("app/artifact_catalog.json")
MODEL_ID = "LATENT_DISTRIBUTION_HGB_FIVE_SEED_MEAN"
COHORT_NAME = "final evaluation cohort with disclosed historical baseline access"
SEEDS = [42, 43, 44, 45, 46]
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024

import sys
from pathlib import Path

# Make `import llm_lab_utils` resolve to the Kaggle dataset copy, exactly as on Kaggle.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets" / "llm-lab-utils"))

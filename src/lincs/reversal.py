from pathlib import Path
import pandas as pd

LINCS = Path("data/external/lincs")
LINCS.mkdir(parents=True, exist_ok=True)
MOD = Path("modeling/modules")
MOD.mkdir(parents=True, exist_ok=True)


def load_l1000_metadata():
    # TODO: read metadata file and filter cell_line == "HA1E"
    return pd.DataFrame()


def compute_connectivity(injury_signature: pd.Series) -> Path:
    # TODO: cosine / tau score vs HA1E perturbation signatures
    out = MOD / "lincs_reversal_scores.csv"
    pd.DataFrame({"drug": [], "score": []}).to_csv(out, index=False)
    return out

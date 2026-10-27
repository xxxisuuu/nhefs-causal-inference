"""
load + clean NHEFS (NHANES I Epidemiologic Followup Study)
data from the causaldata R pkg via Rdatasets mirror. same data as Hernan & Robins "What If" book
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
# pinned to a commit so a later change upstream cannot silently change the results
URL = ("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/"
       "cc03c29690889dbe83089f5206e2422db8c3f71f/csv/causaldata/nhefs.csv")

# confounders, same set as ch.12 of the book
CONFOUNDERS = ["sex", "race", "age", "education", "smokeintensity", "smokeyrs",
               "exercise", "active", "wt71"]


def load(download_if_missing=True):
    f = DATA / "nhefs.csv"
    if not f.exists():
        if not download_if_missing:
            raise FileNotFoundError(f)
        print("downloading nhefs.csv ...")
        pd.read_csv(URL).to_csv(f, index=False)
    d = pd.read_csv(f)
    d = d.drop(columns=["rownames"], errors="ignore")

    # censored = no 1982 weight (63 ppl). keep them for the censoring model
    d["cens"] = d["wt82"].isna().astype(int)

    # categorical -> make sure they're category dtype for the formulas
    for c in ["education", "exercise", "active"]:
        d[c] = d[c].astype("category")
    return d


if __name__ == "__main__":
    d = load()
    print(d.shape)
    print(d.groupby("qsmk")[["age", "wt71", "smokeintensity", "wt82_71"]].mean().round(2))
    print("censored:", d.cens.sum())

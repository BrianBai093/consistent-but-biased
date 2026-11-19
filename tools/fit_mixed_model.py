"""Sensitivity analysis 0 of plan v1.0: logistic models of "operate" on arm.

Needs statsmodels, which is not a project dependency; run it as

    uv run --with statsmodels --with pandas python -m tools.fit_mixed_model

Two fits, both with the neutral arm as reference:

- a random-intercept logistic model per case (statsmodels' variational Bayes
  binomial mixed GLM -- the frequentist GLMM is not in statsmodels), reported
  as posterior means and standard deviations on the log-odds scale;
- a GEE logistic model with exchangeable correlation within case, as a
  population-averaged check that needs no random-effect distribution.

An arm with almost no events (the avoidant arm operated once in 60 runs)
gives an unstable, very negative coefficient; that is reported, not fixed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from tools.analyze_text_experiment import ARMS, load


def frame(out: Path, manifest: Path, attempts: list[int]) -> pd.DataFrame:
    cases = load(out, json.loads(manifest.read_text(encoding="utf-8")), attempts)
    rows = [
        {"case": case.uid, "arm": arm, "operate": int(run.decision == "operate")}
        for case in cases
        for arm in ARMS
        for run in case.runs.get(arm, [])
    ]
    data = pd.DataFrame(rows)
    data["arm"] = pd.Categorical(data["arm"], categories=list(ARMS))
    return data


def main() -> int:
    data = frame(
        Path("runs"),
        Path("data/test_sets/pmc_dcm_v1.json"),
        [1, 2, 3, 4],
    )
    formula = "operate ~ C(arm, Treatment('neutral'))"
    mixed = BinomialBayesMixedGLM.from_formula(formula, {"case": "0 + C(case)"}, data)
    fitted = mixed.fit_vb()
    gee = sm.GEE.from_formula(
        formula,
        groups="case",
        data=data,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    result = {
        "runs": len(data),
        "cases": data["case"].nunique(),
        "mixed_vb": {
            name: {"mean": float(mean), "sd": float(sd)}
            for name, mean, sd in zip(
                mixed.fep_names, fitted.fe_mean, fitted.fe_sd
            )
        },
        "case_sd_log": float(fitted.vcp_mean[0]),
        "gee": {
            name: {
                "coef": float(gee.params[name]),
                "se": float(gee.bse[name]),
                "p": float(gee.pvalues[name]),
            }
            for name in gee.params.index
        },
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

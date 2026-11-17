# Model Card: Home Credit Default Risk Score

This README introduces `Model_Card_Home_Credit.ipynb`, the model card for the XGBoost default-risk score built in `Modeling_Home_Credit.ipynb`. A model card is a plain-language record of what a model is, what it is for, how well it works, and where it can go wrong. This one is written for decision-makers, risk analysts, and compliance reviewers.

## Files

| File | What it is |
|---|---|
| `Model_Card_Home_Credit.html` | **Start here.** The compiled model card: narrative, tables, and figures, with no code shown |
| `Model_Card_Home_Credit.ipynb` | The executed notebook behind it. All code cells are collapsed or hidden; the outputs are saved in the file |

The notebook re-uses the final trained model (`final_model.joblib`) and the prepared data from the modeling notebook, so it does not retrain from scratch. It refits the same settings once on 80% of the training data to get honest holdout predictions, and fits a second copy without protected and proxy fields for the fairness comparison. Run time is about one minute.

## How the card is organized

The executive summary comes first, because most senior readers will stop there. The sections that follow match the requested outline:

1. **Model Details:** model type (XGBoost classifier), version 1.0, training data summary, key hyperparameters.
2. **Intended Use:** who uses it, which decisions it informs, and what it is *not* designed for.
3. **Performance Metrics:** AUC (cross-validated, holdout, and Kaggle), plus precision and recall at the chosen cutoff.
4. **Decision Threshold Analysis:** sourced cost assumptions, the profit-maximizing cutoff, and a sensitivity analysis.
5. **Explainability:** SHAP on a 1,000-row sample, and the top predictive features.
6. **Adverse Action Mapping:** plain-language denial reasons for the top features, with worked examples.
7. **Fairness Analysis:** approval rates by gender and education at the chosen cutoff.
8. **Limitations and Risks:** known weaknesses, likely failure points, and missing data.
9. **References.**

## Summary of the findings

**Performance.** Ranking quality (AUC) is 0.757 in cross-validation and 0.760 on the holdout. On Kaggle it is 0.7445 (public) and 0.7451 (private). Plan with the Kaggle figure, which is about 0.014 below the holdout.

**Cost assumptions and threshold.** The card uses an 8% profit on a repaid loan and a 70% loss on a defaulted loan (a 30% recovery rate). They are anchored in Home Credit B.V.'s reported 2018 net interest margin (15.8%) and cost-to-income ratio (42.2%), a Fitch recovery-rate assumption of 30% for unsecured consumer loans, a reported average recovery of about 35% on Dutch unsecured consumer loans, and an industry range of 60% to 90% for loss given default. Sources are listed in section 9. Approving a loan pays off when the predicted risk is below profit ÷ (profit + loss) = 8% ÷ 78% = **10.3%**. The predicted risks match actual default rates by decile, and a check on the holdout, using real loan amounts, puts the profit-maximizing cutoff at 10.0%.

**At the 10.3% cutoff (holdout results):**

| Measure | Value |
|---|---|
| Applicants flagged for decline or review | 25% |
| Approval rate | 75% |
| Default rate among approved loans | 4.4% (vs. 8.1% for everyone) |
| Precision / recall for predicting default | 19% / 59% |
| Expected profit per 100,000 applications | 2,171 vs. 1,262 million currency units if everyone were approved (+1.5% of loan volume) |
| Gain across nine cost scenarios (profit 4% to 12%, loss 50% to 90%) | -0.03% to +3.1% of loan volume |

The optimal cutoff itself ranges from about 4% to 19% across those scenarios, so the cutoff should be recomputed with Home Credit's own margin and recovery data before a pilot.

**Explainability.** Outside credit-score data (`EXT_SOURCE_*`) is by far the strongest driver, followed by loan size and payment, then employment and car ownership. Gender, education, and family status are also among the top 12 features.

**Adverse action.** Each top feature is mapped to a plain-language reason (for example, low external scores become "Low score from external credit-data sources", and a missing first external score becomes "Limited external credit history on file"). Each is marked *Usable*, *Legal review*, or *Do not cite*. Gender, education, and family status cannot be cited, which is a strong reason to remove them from the model.

**Fairness.** At the chosen cutoff, women are approved at 80% and men at 66% (ratio 0.82), and approval ranges from 66% (lower secondary) to 87% (higher education) by education (ratio 0.76, below the 0.80 screening line). Part of this reflects real differences in default rates, but the model uses gender directly. A version without the protected and proxy fields narrows the gaps (ratios 0.90 and 0.82) at a cost of about 0.007 AUC and 1.2% of profit, but does not eliminate them.

## Recommendation and key caveats

Use the score as a ranking tool in a shadow run and then a pilot, with flagged applications going to human review. **Remove gender and the other protected or proxy fields before any live use.**

- The savings come from approved loans only (the data contain no declined applicants) and rest on assumed profit and loss rates, so the real gain over the current process is probably smaller than shown.
- Thin-file applicants were not evaluated separately, and the supplementary tables (bureau, previous loans, payment history) are not used.
- Only gender and education were checked for fairness; age, family status, region, and combinations still need review with fair-lending counsel.
- About 4 in 5 flagged applicants would actually have repaid, so a flagged applicant should be reviewed, not automatically declined.

## Reproducing it

Place `Model_Card_Home_Credit.ipynb`, `Data_Prep.py`, `final_model.joblib`, and the folder `Home Credit Data/prepared/` (created by the modeling notebook) in one folder. You need `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `scipy`, `matplotlib`, and `joblib`. To get the code-free version, run: `jupyter nbconvert --to html --no-input --no-prompt Model_Card_Home_Credit.ipynb`.

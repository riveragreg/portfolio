# Home Credit Default Risk: Modeling

This project builds a repayment-risk score for Home Credit loan applications. The score ranks applicants by their chance of payment difficulty (`TARGET = 1`, about 8% of applications), so that underwriters can route applications to automatic approval, automatic decline, or manual review.

This README introduces the modeling notebook, `Modeling_Home_Credit.ipynb`: which models I tried, how they performed, and why I chose the final model. The data-preparation script is described in `README_Data_Prep.md`.

## Files

| File | What it is |
|---|---|
| `Modeling_Home_Credit.ipynb` | The modeling notebook (executed, with outputs) |
| `Data_Prep.py` | Reusable data-preparation functions that implement the EDA decisions (see `README_Data_Prep.md`) |
| `EDA_Home_Credit.ipynb` | The exploratory analysis the preparation and modeling decisions come from |
| `submission.csv` | Predicted default probabilities for the 48,744 test applications, in Kaggle format |

## How to run

1. Put `Modeling_Home_Credit.ipynb` and `Data_Prep.py` in one folder, and the Kaggle files `application_train.csv`, `application_test.csv`, and `sample_submission.csv` in a subfolder called `Home Credit Data`.
2. Install the packages: `pip install pandas numpy scikit-learn scipy matplotlib joblib lightgbm xgboost imbalanced-learn`.
3. Run all cells. On a single-CPU machine the full notebook takes about 21 minutes (model comparison about 9 minutes and the class-imbalance experiment about 7). The prepared tables are cached in `Home Credit Data/prepared`, so later runs skip the preparation step.

Inputs are the three CSV files above (307,511 training rows and 48,744 test rows). The notebook does not use the optional supplementary tables (bureau, previous applications, payment histories).

## What the notebook does

1. **Prepare the data** with `Data_Prep.py`: 82 candidate features (69 numeric, 13 categorical), including 24 engineered features, missing-data indicators, and capped financial ratios. All thresholds are learned from the training data only and re-used on test.
2. **Set a benchmark.** Majority-class classifier and a one-variable logistic regression.
3. **Compare eight models** with 3-fold cross-validation on all development rows (80% of the training data). The other 20% is a holdout that is not used for any decision until the final check.
4. **Test class-imbalance fixes** (class weights, down-sampling, up-sampling, SMOTE) against no adjustment.
5. **Tune the best model** with randomized search (20 settings, 3-fold CV) on 5,000-row and 25,000-row subsamples, and choose the final settings on the holdout.
6. **Report performance and run time**, show how the score would route applicants, and write the Kaggle submission.

I use **AUC** as the main metric. Accuracy is not useful here: a model that predicts "nobody defaults" is 91.9% accurate.

## Models tried and how they performed

Cross-validated AUC (3 folds, 246,008 development rows; the standard deviation across folds is about 0.001):

| Model | Predictors | CV AUC | Time for 3 folds |
|---|---|---|---|
| Majority class | none | 0.5000 | under 1 second |
| Logistic regression (benchmark) | `EXT_SOURCE_MEAN` only | 0.7164 | 1 second |
| **XGBoost** | all features | **0.7568** | 47 seconds |
| LightGBM | all features | 0.7552 | 39 seconds |
| Logistic regression | all features | 0.7500 | 17 seconds |
| XGBoost | all minus high-governance fields | 0.7489 | 29 seconds |
| LightGBM | all minus high-governance fields | 0.7471 | 31 seconds |
| Random forest | all features | 0.7454 | 379 seconds |
| Logistic regression | all minus high-governance fields | 0.7416 | 8 seconds |
| Logistic regression | EDA baseline set | 0.7375 | 5 seconds |

- Accuracy is about 0.919 for every model, so it cannot separate them. AUC does.
- Gradient boosting beats logistic regression by about 0.005-0.007 AUC, and the random forest is both lower and much slower.
- A small, explainable feature set (the EDA baseline) already reaches 0.7375, mostly from the external scores.
- The **high-governance fields** (education, family status, housing, occupation, organization, region, gender, age) add only about 0.008 AUC for each model type.

**Class imbalance.** No adjustment gave the best AUC for both XGBoost (0.7568) and logistic regression (0.7500). Class weights were nearly tied (0.7562 and 0.7498); down-sampling, up-sampling, and SMOTE were lower. All of them also lowered accuracy at a 0.5 cutoff (to about 0.68-0.71) and inflated the predicted probabilities. The final model trains on the real default rate.

**Tuning.** Tuning on subsamples improved the subsample cross-validation score, but the tuned settings did worse on the full-data holdout than the default settings:

| XGBoost settings | Holdout AUC |
|---|---|
| Default (300 trees, learning rate 0.05, depth 4) | **0.7596** |
| Tuned on 25,000 rows | 0.7544 |
| Tuned on 5,000 rows | 0.7495 |

Small samples favor shallow, heavily regularized trees, which underfit when there are 246,000 rows.

## Final model and why I chose it

**Final model: XGBoost on all features with its default settings, trained on all 307,511 training applications.**

- **It had the best cross-validated AUC** of the eight candidates (0.7568), ahead of LightGBM (0.7552) and logistic regression (0.7500), with fold-to-fold variation of only about 0.001.
- **Its settings were chosen on data it had not seen.** The default settings beat both tuned versions on the 20% holdout.
- **Performance:** holdout AUC **0.760** (train 0.778, cross-validation 0.757). The holdout agrees with the cross-validation, and the train-to-holdout gap of about 0.019 is a modest amount of overfitting.
- **Run time:** fitting on all training data takes about 26 seconds, and scoring the 48,744 test applications takes about 0.3 seconds.
- **It ranks risk usefully.** On the holdout, the default rate rises from 1.2% in the safest decile to 27.6% in the riskiest (3.4 times the average). Flagging the riskiest 20% of applicants catches 52% of all defaults, and the safest half of applicants has a default rate of 2.8%. The predicted probabilities are also close to the actual rates in each decile.

**Kaggle score.** Was submitted `submission.csv` and record the public and private scores in the `KAGGLE_PUBLIC_SCORE` and `KAGGLE_PRIVATE_SCORE` as 0.74454 and 0.74509 accordingly.

## Important caveats

- **The competition model is not a production model.** The best-AUC model uses gender and education, among others. The feature report says gender must not be used as a repayment-risk input. Removing the governed fields costs only about 0.008 AUC (XGBoost 0.7489 versus 0.7568), so for the lender I recommend the version without them, and a fairness check across groups before any use.
- **Explainability.** Logistic regression on all features is only about 0.007 AUC behind XGBoost and is the easiest model to turn into plain adverse-action reasons.
- **Application data only.** The supplementary tables are the most likely route to a better score, and `Data_Prep.py` already supports them. The BPS goal of approving more thin-file applicants has not been tested here, and `TARGET` exists only for applications that became loans.
- **AUC measures ranking only.** Measuring the BPS metrics (expected credit loss, approval rate, 90+ day delinquency) needs approval cutoffs based on the cost of a default and the profit from a good loan. That is the next phase.

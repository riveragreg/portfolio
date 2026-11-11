# Home Credit Default Risk: A Risk Score That Pays for Itself

**In one line:** I built a credit-risk score for Home Credit loan applications that ranks applicants by their chance of payment trouble, and showed that acting on it could raise profit by about 1.5% of loan volume while cutting the default rate among approved loans from 8.1% to 4.4%.

## 1. The business problem and objective

Home Credit lends to people with little or no credit history, so standard bureau data is often missing for them. Every decision risks one of two mistakes: approving someone who later defaults (a credit loss), or declining someone who would have repaid (lost growth and a customer who deserved credit). About 8% of the 307,511 historical loans in the training data ran into payment trouble.

**Objective:** build a repayment-risk score that helps underwriters route each application to automatic approval, manual review, or decline, to cut losses without simply tightening approvals, and to do it in a way that is explainable and fair.

**Talking points:** how I framed the problem around business metrics (expected loss, approval rate, review volume) instead of a model score alone.

## 2. The solution

| Stage | What I did | Key result |
|---|---|---|
| **EDA** (`EDA_Home_Credit.ipynb`) | Nine questions traced to the business problem statement, the data map, and a lending feature report | The strongest signal was outside credit scores; the affordability ratios I expected to matter were weak |
| **Data preparation** (`Data_Prep.py`) | Reusable functions that apply the EDA decisions; every threshold is learned on train only and reused on test | 24 engineered features; identical train and test columns, one row per applicant |
| **Modeling** (`Modeling_Home_Credit.ipynb`) | Benchmark, 8 candidate models with cross-validation, class-imbalance tests, randomized-search tuning, holdout check | XGBoost on all features: AUC 0.757 cross-validated, 0.760 holdout, **0.745 on Kaggle** |
| **Model card** (`Model_Card_Home_Credit.html`) | Cost-based decision threshold, SHAP explanations, adverse-action reasons, fairness checks, limitations | Recommended cutoff: decline or review at 10.3% predicted risk or higher |
| **Presentation** | 8-minute business story with a customer persona | Recommendation: 3-month shadow run, then a pilot |

**Key decisions:** the final model uses XGBoost's default settings, because settings tuned on small samples did worse on the full data. No class-imbalance fix helped, so I train on the real default rate. I recommend a production version without gender and other protected or proxy fields.

**Talking points:** why AUC and not accuracy; why a cost-based threshold; why the best-scoring model is not the one I recommend.

## 3. The business value

At the cost-based cutoff, on 61,503 held-out applications:

- **Default rate among approved loans falls from 8.1% to 4.4%**, while 75% of applicants are still approved.
- **Expected profit rises about 72%**: from 1,262 to 2,171 million currency units per 100,000 applications, or **+1.5% of loan volume**.
- **The cutoff catches 59% of the applicants who go on to have trouble** while flagging 25% of all applicants for review or decline.
- **The result holds up under uncertainty.** Across nine cost scenarios the gain ranges from about 0% to +3.1% of loan volume, so it is never materially negative.
- **Fairness has a small price.** Dropping gender and the other protected fields costs about 0.007 AUC and 1.2% of profit.

**Caveats I would raise myself:** the data contain only approved loans, so the real gain over the current process is probably smaller; the profit and loss rates are sourced assumptions to replace with Home Credit's own figures; and thin-file applicants were not tested separately.

## 4. Difficulties along the way

**Shaping the EDA questions.** The hardest early step was deciding what to look at beyond "who will default". Starting from the business problem statement and company information, I asked what else the lender needs to know. Are affordability ratios actually predictive? What do the missing values mean (an 18% placeholder in employment dates, a missing first outside score for 56% of applicants)? Which fields are risky to use? That produced nine traceable questions and changed my assumptions: the strongest signals were also the riskiest to use, and the outcome in the data (payment difficulty) is not the same as the 90+ day delinquency the business tracks.

**Putting a dollar value on the model with no cost data.** The dataset has no profit, loss, or recovery figures. I sourced them from Home Credit's reported margins and published recovery rates for unsecured consumer loans (Fitch and others), turned them into a cutoff (profit ÷ (profit + loss) = 8% ÷ 78% = 10.3%), and showed how the result changes under other assumptions. I was explicit about where my numbers were approximations.

**Accuracy is misleading when only 8% default.** "Approve everyone" is already 91.9% accurate, so accuracy could not tell the models apart. I switched to AUC for ranking, and to precision, recall, and expected profit at a chosen cutoff. The usual class-imbalance fixes (class weights, SMOTE, up- and down-sampling) did not improve ranking and distorted the probabilities, so I did not use them.

## 5. What I learned

- **Working with an AI coding agent in a real repository.** I used ChatGPT Codex inside VS Code to commit and implement changes in a GitHub repository, which made changes traceable and reviewable.
- **Completing a data-science project with a stage-gated AI workflow.** I moved from business problem to EDA, data preparation, modeling, and communication in stages, with each stage reviewed before the next one began.
- **Structuring a project around the main data while leaving room to grow.** The pipeline runs on the main train and test files, and I wrote and tested separate functions for the related CSVs (bureau, previous applications, payment histories) that can be switched on later to try to improve accuracy. I have not run them on the real supplementary files yet.

## Next steps

Add the supplementary-table features and re-measure; remove protected fields and re-check fairness; evaluate thin-file applicants separately; replace the assumed costs with Home Credit's own; then run a shadow period and pilot.

## Files

| File | What it is |
|---|---|
| `EDA_Home_Credit.ipynb` | Exploratory analysis and the decisions it produced |
| `Data_Prep.py` | Reusable preparation functions ([details](README_Data_Prep.md)) |
| `Modeling_Home_Credit.ipynb` | Model comparison, tuning, final model, Kaggle submission ([details](README_Modeling.md)) |
| `Model_Card_Home_Credit.html` | Model card for decision-makers ([details](README_Model_Card.md)) |
| `submission.csv` | Kaggle predictions (public 0.7445, private 0.7451) |

# Text Analysis for Charge-off prediction

## Objectives

1. Predict binary loan outcomes (charged-off vs. paid-off).
2. Measure the incremental value of text features from the free-form loan description.

## Evaluation metric

Charge-offs make up roughly 14% of the data, so a model that always predicts
repayment reaches 86% accuracy. **Recall** on charged-off loans is used instead,
since screening out uncreditworthy borrowers is the costly error to avoid
(Stiglitz and Weiss, 1981).

## Approach

- **Preprocessing:** dropped columns with missing values; set rare events
  (bankruptcies, collections) to 0, then excluded them for low variance;
  winsorized income, account counts and revolving balance at the 1st/99th percentiles.
- **Financial features:** removed identifiers and highly collinear columns, keeping
  core credit and borrower attributes (DTI, loan amount, loan purpose, employment length).
- **Text features:** built bigrams from loan descriptions following Netzer, Lemaire
  and Herzenstein (2019) and scored each by its odds of default,
  P(bigram | defaulted) / P(bigram | repaid). Candidates (the 200 most frequent
  bigrams plus the 50 highest- and lowest-odds bigrams) were ranked with a
  text-only decision tree, and the top 50 were kept.
- **Models:** logistic regression, random forest and XGBoost.

## Text insights

Bigrams with the highest odds of default relate to family (*husband worked*,
*emergency family*), uncertainty (*free worry*, *like stress*) and absolutes
(*always stay*, *cards always*). Those with the lowest odds use specific credit
language and action verbs (*rate good*, *save enough*, *reduce overall*).

![Bigrams with highest odds of default](figures/chgoff_bigrams.jpg)

![Bigrams with lowest odds of default](figures/paid_bigrams.jpg)

## Results

| Model | Recall | Recall gain from text features |
|---|---|---|
| Random forest | **82.4%** | +1.2% |
| XGBoost | 62.0% | +2.0% |
| Logistic regression | 54.0% | — |

Random forest performed best overall. XGBoost gained more from text features,
which suggests gradient boosting may make better use of alternative data.

## Repository

| File | Contents |
|---|---|
| `lending_club_EDA.ipynb` | Data cleaning, exploratory analysis and bigram analysis |
| `lending_club_modeling.ipynb` | Feature engineering, model training and evaluation |
| `figures/` | Bigram charts, ROC curves and confusion matrices |

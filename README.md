# From Churn Prediction to Intervention

This repository contains the model-training core reported in the manuscript.

## Included

1. `train_reference.py` compares ridge regression, gradient boosting, and random forest on a user-level 80/20 split of retained users. Spearman correlation selects the contextual seven-day comment reference.
2. `disappointment.py` computes the signed post-level gap (`reference - observed comments`) and the user-level trajectory summaries described in the manuscript.
3. `train_churn.py` evaluates the same LightGBM learner with and without the trajectory features over the same five user-level folds.

## Prepared inputs

`train_reference.py` expects one row per Set A post. Required columns are `user_id` and `comments_7d`; every other numeric column is treated as a publication-described reference-model feature.

`disappointment.py` expects one row per Set B post with `user_id`, `post_time`, `observed_comments`, and `expected_comments`.

`train_churn.py` expects one row per Set B user with `user_id`, `churned`, the original churn features, and the trajectory columns produced by `disappointment.py` (all prefixed `disappointment_`).

```bash
python train_reference.py set_a_posts.csv reference_model.joblib
python disappointment.py set_b_posts_with_references.csv trajectories.csv
python train_churn.py set_b_user_features.csv metrics.csv
```

Raw-data preprocessing and the LLM intervention experiment are outside this release because they are not part of the model-training core requested for publication.


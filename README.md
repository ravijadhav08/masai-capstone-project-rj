# Delhivery Last-Mile Delay Intelligence Pipeline

Capstone project for the Artificial Intelligence and Machine Learning Certification
Program (E&ICT Academy, IIT Roorkee). One connected repository answers three
operations questions: how long a parcel will really take (regression), which
parcels will miss their promise (classification + alert threshold), and which
hubs need which playbook (K-Means segmentation). A final scoring script turns the
48 new parcels into a concrete action list.

All data is synthetic, produced by `generate_data.py`; the Delhivery brand is only
the setting for the decisions.

## How to run

From a fresh clone, with Python 3.12 (any recent 3.x works):

```bash
pip install -r requirements.txt
python generate_data.py        # run once; must print: 24 1224 48
python part_a_clean_eda.py     # writes data/modelling_table.csv + Part A outputs
python part_b_regression.py    # Part B regression comparison
python part_c_classification.py# Part C classifier, tuning, alert threshold
python part_d_segments.py      # Part D hub segmentation + tier check
python package_model.py        # writes models/ artefacts + reload check
python predict_delay.py        # scores the 48 new parcels -> action list
```

`requirements.txt` pins the four packages with `==` to the exact versions used
(pandas 3.0.3, numpy 2.4.6, scikit-learn 1.9.1, joblib 1.6.0). The committed
`models/delay_model.pkl` loads reliably only on the scikit-learn version pinned in
`requirements.txt`; **running `python package_model.py` regenerates
`models/delay_model.pkl`** from the CSV configuration, so when the installed
versions differ, re-run `package_model.py` instead of loading the committed pickle.
Every script runs top to bottom offline and uses only files in the repository;
`predict_delay.py` produces byte-identical output on consecutive runs.

## Data findings

All numbers below are computed by `part_a_clean_eda.py` and appear in
`outputs/part_a_checks.csv`, `outputs/part_a_delay_by_group.csv` and
`outputs/part_a_correlations.csv`.

1. **Express is the service that breaks its promise most often.** The delay rate
   is 0.4897 for Express (119 of 243 shipments), versus 0.2056 for Standard
   (132 of 642) and 0.0857 for Economy (27 of 315). Express promises 36 hours
   against a base flow that already averages ~37 hours, so even normal noise
   pushes it past the promise; Economy at 96 hours almost never slips.
2. **Delay rate climbs steeply with hub tier:** 0.04 at Tier 1 (16 of 400),
   0.215 at Tier 2 (86 of 400) and 0.44 at Tier 3 (176 of 400). Tier 3 hubs run
   longer lanes (mean distance rises per tier) and carry the stretched
   operations profile. Overall, 278 of 1,200 shipments were delayed
   (delay rate 0.2317).
3. **Distance is the strongest operational driver.** Across distance bands the
   delay rate rises from 0.0219 (0–200 km, 6 of 274) to 0.1368 (201–400 km,
   77 of 563), 0.4364 (401–600 km, 120 of 275) and 0.8523 (601+ km, 75 of 88).
   More than 8 in 10 parcels on lanes over 600 km arrive late.
4. **The three columns most strongly correlated with `delivery_hours`** (ranked
   by absolute correlation, excluding the target itself) are `promised_hours`
   at 0.8372, `distance_km` at 0.4013 and `n_stops` at 0.3247 — the service
   promise, the lane length and the rider-route length dominate delivery time.
5. **`declared_value_inr` was log-transformed because it is strongly
   right-skewed:** on the median-filled column the mean is INR 1,533.05 while the
   median is INR 1,121.50 — a 37% gap. After the natural log the mean is 7.0277
   and the median 7.0224, nearly equal, so the transformed feature no longer
   lets a few high-value parcels dominate the model.

Cleaning evidence (`part_a_checks.csv`): the raw export has 1,224 rows;
24 exact duplicates were removed, leaving 1,200 unique shipments. Before any
filling there were 36 missing `distance_km`, 24 missing `parcel_weight_kg`
(the 5 scanner-fault sentinel weights of 999 kg are counted separately), 12
missing `declared_value` and 18 missing `payment_mode`; `service_type` appeared
in 12 spellings and was normalised to 3. After the hub merge 1,200 rows remain
and the modelling table has 0 missing cells across the 17 feature columns plus
the target columns. The median fill values are 314.95 km, 1.83 kg and
INR 1,121.50. `delivery_hours` has one statistical outlier under the
1.5 x IQR rule (bounds -0.7063 to 116.5238), which was kept.

## Model results (regression)

Inputs are the 17 pinned feature columns; the target is `delivery_hours`. The
1,200 rows were split 840 / 180 / 180 (train / validation / test) with
`random_state=42`. The leakage check in `part_b_checks.csv` shows the scaler was
fitted on training rows only: the largest absolute column mean after scaling is
0.0 on the training rows but 0.2187 on the test rows.

* **Baseline.** Predicting the training mean for every validation parcel gives
  R2 = -0.0212 (`part_b_metrics.csv`): the model explains nothing and is
  slightly worse than the validation-set mean, exactly as a useless predictor
  should be.
* **Winner on validation: `linear_regression`** with validation RMSE 4.8912
  hours and R2 = 0.9313; scored once on the test rows it reaches RMSE 4.5235
  hours and R2 = 0.9493. Gradient boosting is a close second (validation RMSE
  4.9165, R2 = 0.9306), followed by random forest (RMSE 5.468, R2 = 0.9141) and
  the depth-6 decision tree (RMSE 6.2796, R2 = 0.8867). The underlying
  relationship is close to additive, so a regularised linear model matches the
  ensembles.
* **Bias-variance picture** (`part_b_tree_depth.csv`): depth 1 is high-bias —
  train RMSE 11.6334 and validation RMSE 12.1324 are both high (gap 0.499).
  The unlimited tree is high-variance — train RMSE 0.0 against validation RMSE
  7.789 (gap 7.789): it memorises the training set and degrades on new data.
  Depth 6 has the lowest validation RMSE (6.2796); going deeper only narrows
  the training error while validation error rises.
* **Learning rate** (`part_b_gb_learning_rate.csv`): 0.1 is best
  (validation R2 = 0.9306). At 0.01 the boosting learns too slowly — 100
  rounds are not enough and R2 stays at 0.7127. At 1.0 each tree overshoots
  the residual and the model overfits, dropping to R2 = 0.844.
* **Cross-validation vs validation.** Five-fold CV on the 1,020 training-plus-
  validation rows (`part_b_cv.csv`) ranks gradient boosting first
  (mean R2 = 0.9352, std 0.0044) ahead of linear regression (mean R2 = 0.9261,
  std 0.0048) — the opposite of the validation ranking. I would still ship the
  **linear regression**: it won on untouched validation data, the CV gap
  (0.009) is within fold-level noise for a 1,020-row training pool, it is
  simpler to explain to planners, and it trains and scores in milliseconds.
  Both models comfortably exceed the R2 >= 0.88 bar on test.

## Model results (classification)

Inputs are the same 17 columns; the target is `is_delayed` (23.17% of parcels).
The stratified split keeps the delayed share in every fold: 195 / 42 / 41
delayed parcels in train / validation / test.

* **Blind baseline.** Predicting "on time" for every parcel scores accuracy
  0.7667 on validation (TN 138, FP 0, FN 42, TP 0) yet is operationally useless:
  it never makes a call, so recall is 0 and all 42 delayed parcels are missed
  (`part_c_models.csv`).
* **Noise columns.** The five bookkeeping columns (`label_print_batch`,
  `scanner_build_no`, `tape_roll_no`, `sticker_serial_mod`, `shelf_slot_no`)
  together hold 0.2245 of the default forest's feature importance
  (`part_c_feature_scores.csv`) despite carrying no signal — pure chance on a
  finite training set. All five sit outside the top 7 by F-score (ranks 8–17,
  each F-score below 10), while every one of the top seven real features has an
  F-score above 30 (the lowest is `sale_event_flag` at 36.7281). The top seven
  are exactly `distance_km`, `hub_tier`, `hub_daily_capacity`, `n_stops`,
  `promised_hours`, `rain_flag`, `sale_event_flag`.
* **Choosing k.** Validation F1 by number of selected features
  (`part_c_selectk.csv`): k = 5 gives 0.6988, **k = 7 gives 0.8333**,
  k = 10 gives 0.7857, k = 12 gives 0.8049 and k = 17 gives 0.7848. Keeping the
  seven real signals and dropping the noise beats using all columns, so k = 7
  is used from there on.
* **Candidate models** (`part_c_models.csv`): logistic regression reaches F1
  0.8046 (accuracy 0.9056), the best depth-6 tree F1 0.747 (accuracy 0.8833)
  and the default random forest F1 0.7848 (accuracy 0.9056) — every trained
  model beats the blind baseline's accuracy 0.7667.
* **Tuning did not beat the untuned forest.** On the 7 selected features the
  untuned forest (100 trees, unlimited depth) scores validation F1 = 0.8333,
  ahead of randomized search (0.7848; best CV accuracy 0.8905) and grid search
  (0.775; best CV accuracy 0.8952) over 27 and 9 candidates (`part_c_tuning.csv`).
  With only 180 validation rows, the searches pick parameters that maximise
  cross-validated accuracy on 840 training rows, and that advantage does not
  transfer — evidence that aggressive tuning on a small validation set mostly
  harvests noise. The final configuration is therefore the default forest on
  the 7 selected features (`part_c_final_config.csv`: selector_k = 7,
  n_estimators = 100, max_depth = unlimited, min_samples_leaf = 1,
  chosen_from = default_forest_same_7_features).
* **Alert threshold.** Because a missed delayed parcel costs more than an
  unnecessary call, the threshold is the highest value whose validation recall
  is at least 0.80 (`part_c_thresholds.csv`). Recall falls from 0.9762 at 0.1
  through 0.8333 at 0.5 to 0.7619 at 0.6, so the chosen threshold is **0.5**,
  with validation precision 0.7955, recall 0.8333 and F1 0.814. The trade-off:
  one in five flagged parcels is a false alarm, but only about one in six
  delayed parcels escapes uncalled.
* **Single test evaluation** (`part_c_final_test.csv`, threshold 0.5, evaluated
  exactly once): TN 125, FP 14, FN 10, TP 31 — precision 0.6889,
  recall 0.7561, F1 0.7209, accuracy 0.8667. Test recall (0.7561) sits below
  validation recall (0.8333); that gap should be read honestly — the test set
  contains only 41 delayed parcels, so each missed parcel moves recall by about
  2.4 points, and 10 were missed here.

## Operations recommendation

* **Escalate segment (9 hubs, mean delay rate 0.46, `segment_profile.csv`).**
  These hubs get the hub-manager playbook: Guwahati 0.62, Agartala 0.54,
  Raipur 0.48, Nagpur 0.46, Patna 0.46, Siliguri 0.44, Bhubaneswar 0.40,
  Ranchi 0.38, Kochi 0.36 (`hub_profile.csv`). None of the nine runs better
  than a 36% delay rate.
* **Tier labels that lie** (`tier_vs_segment_mismatch.csv`): HUB07 Kolkata is
  labelled Tier 1 but behaves Watch; HUB13 Nagpur and HUB15 Kochi are labelled
  Tier 2 but behave Escalate; HUB22 Dehradun is labelled Tier 3 but behaves
  Watch. Treat the tier label as a planning hint, not a verdict: Kolkata should
  be resourced like a Watch hub, Nagpur and Kochi like Escalate hubs, and
  Dehradun can be spared the most expensive Tier 3 interventions.
* **What the alert threshold buys.** At the 0.5 threshold, about 7 of every 10
  flagged parcels are really delayed (test precision 0.6889) and about 7.6 of
  every 10 delayed parcels are caught (test recall 0.7561,
  `part_c_final_test.csv`).
* **The 48 new parcels** (`new_shipment_actions.csv`): 8 were flagged
  (delay_probability >= 0.5). Of those, 6 sit in Escalate-segment hubs and get
  "Escalate to hub manager"; the other 2 get a "Proactive customer call"; the
  remaining 40 need no action.
* **Why K = 3 stays.** The across-to-within distance ratio rises with K —
  2.894 (K = 2), 5.3928 (K = 3), 6.3047 (K = 4), 6.8866 (K = 5,
  `part_d_separation.csv`) — but a bigger ratio is not a better organisation:
  K = 4 produces a 4-hub cluster and K = 5 a 2-hub cluster (`cluster_sizes`
  8-7-5-4 and 8-7-4-3-2), too small to carry their own playbook. Operations
  runs exactly three playbooks, and K = 3 is also the most stable solution
  (the K = 2 and K = 3 clusterings reproduce under every tested seed and hub
  ordering). The three segments separate cleanly by behaviour: Standard
  (7 hubs: 47.2396 mean hours, 0.0257 delay rate), Watch (8 hubs: 56.8137
  hours, 0.155 delay rate) and Escalate (9 hubs: 64.6504 hours, 0.46 delay
  rate).

## Production plan

**Repository mapping of production-readiness characteristics**

* **Accuracy** — held-out evaluation is baked into the pipeline:
  `part_b_metrics.csv` (test row) and `part_c_final_test.csv` are written by
  code, so accuracy claims are reproducible, not asserted.
* **Reliability** — `part_d_reload_check.csv` proves the shipped artefact works:
  180 test rows scored, reload probability diff 0.0 between the in-memory and
  the reloaded-from-disk pipeline.
* **Reproducibility** — `requirements.txt` pins all four libraries with `==`,
  every randomised step uses `random_state=42`, `generate_data.py` is committed
  verbatim, and `predict_delay.py` is byte-identical across consecutive runs.
* **Performance (speed)** — feature selection keeps 7 of 17 columns
  (`part_c_final_config.csv`), and both shipped models (linear regression,
  100-tree forest) score a 48-parcel batch in milliseconds.
* **Maintainability / consistency** — a single `src/shipment_prep.py` holds the
  cleaning and feature code; training (`part_a_clean_eda.py`) and scoring
  (`predict_delay.py`) call the same functions, and `models/feature_order.csv`
  plus `models/impute_values.csv` force identical column order and imputation
  at predict time — no second copy of the preprocessing can drift out of sync.

**Quantities to monitor, with numeric triggers**

1. **Weekly delayed share of delivered parcels** — historical baseline 0.2317
   (`part_a_checks.csv`); alert if a week's share exceeds 0.26 (baseline + ~3
   percentage points) for two consecutive weeks.
2. **Weekly share of parcels flagged** — baseline on new traffic is 8 of 48
   (~0.167); alert if the flagged share leaves the 0.10–0.35 band, which would
   signal either threshold drift or a change in the delay mix.
3. **Recall on parcels whose outcome is now known** — test baseline 0.7561;
   alert if the trailing-four-week recall falls below 0.70, meaning the alert
   system is missing more than 3 in 10 delayed parcels.
4. **Calibration drift** — alert if the mean predicted probability of delayed
   parcels deviates from the observed delayed share by more than 0.05 in a
   week, so the 0.5 alert threshold keeps its meaning.

**Retraining schedule and event handling.** Refit the full pipeline monthly,
and immediately after any sale event. Weeks inside the 15–28 September sale
window (`sale_event_flag = 1`) must be compared against the same kind of event
window — never against ordinary weeks — because the sale both raises volume and
adds roughly 7 hours (plus 12 more on Express), so an ordinary-week comparison
would look like a massive false alarm. **Data drift** here looks like: hub
delay rates climbing away from `hub_profile.csv` values, the median lane
distance or stops per hub shifting (the Standard/Watch/Escalate separation is
built on those four profile columns), the weekly delayed share persisting above
0.26, or the payment-mode / weekend mix moving far from the training mix —
any of these triggers the retraining run above.

## Limitations

1. **The data is synthetic.** It was produced by `generate_data.py` with
   near-additive structure and bounded noise, so the near-tie between the
   linear model and gradient boosting may not survive contact with real
   logistics messiness (holidays, strikes, address quality).
2. **The test set holds only 41 delayed parcels**, so test recall (0.7561) and
   precision (0.6889) are coarse estimates — each delayed parcel is worth about
   2.4 points of recall — and small changes in the split can move them visibly.
3. **The alert threshold was tuned on one 180-row validation split.** A fresh
   threshold on new data, or a cost-based threshold using real call and
   miss costs, could move the operating point.
4. **Fill values are historical.** The imputation medians (314.95 km, 1.83 kg,
   INR 1,121.50) were computed once on the historical export; if lane lengths
   or parcel mix drift, the fills quietly carry stale information.
5. **Segments are a forced, static K = 3.** Hubs near a segment boundary can be
   mis-bucketed by tiny profile changes, and the model does not know why a hub
   behaves as it does — only that it does.

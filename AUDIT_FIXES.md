# TrafficSense functional review and fixes

Reviewed 20 October 2026, base commit `2d2ebd7de1a9c5733ce3ca2dd46d6ce1e20d012a`.
GitHub was fetched and the remote main commit matched this checkout. All changes
are local; no commits, pushes, or deployment changes were made.

## Scope and verified defects

Inspected the app, both notebooks and their tracked Python exports, requirements,
runtime declaration, README, both CSVs, and both model artifacts. Executed every
code cell in both notebooks, including regeneration of data and artifact saves.
No architecture, feature-set, hyperparameter, or evaluation-split refactor was made.

1. **Incorrect runtime predictions.** Original artifacts were produced with XGBoost
   3.3.0 but requirements pinned 2.1.4. Reproduced MAE 3287.87, R² -1.79184 and 19,743
   negative regression predictions with the original artifacts under 2.1.4. With
   3.3.0, the same original artifacts produced MAE 219.73, R² 0.967598 and 37 negatives.
   Changed `requirements.txt` to 3.3.0; both rebuilt artifacts were saved and reloaded
   under that runtime. Other direct dependency pins are unchanged.
2. **Holiday dates only flagged at midnight.** Preprocessing cell 13 now collects
   known holiday dates before deduplication and flags every observation on those
   dates. Holiday rows increased from 53 to 1,203 over the same 53 dates; exactly
   1,150 processed rows changed, solely in `is_holiday`. Regenerated
   `data/traffic_processed.csv` and both `models/*.pkl` using the original training
   settings and model-selection procedure. No raw CSV changes.
3. **Negative vehicle counts displayed.** `app.py` checks finite regression output
   and clamps it to zero. Training cells 12/14 apply the same clamp when evaluating
   both regression candidates. An actual rebuilt-model prediction of -14.90 for
   2014-01-06 02:00 displays `0 vehicles/hour` in the Streamlit test.
4. **Broken file paths.** App artifacts resolve relative to `app.py`, so launching
   from another folder works. Training cell 23 now creates `models`, matching both
   save destinations, instead of creating `../models`. The actual save cells were
   tested in a fresh directory with no models folder, and artifacts reloaded with
   identical predictions. App loading failures now show an actionable error.
5. **Misleading prediction semantics.** The UI calls the classification result an
   independent volume category estimate, explains possible disagreement, and
   states the 2,274/4,598 boundaries. It no longer calls the result measured
   congestion. Both project prediction tasks and classifier probabilities remain.
   Minneapolis local time and the manual holiday override are identified.

Notebook code changes are mirrored in `.virtual_documents`. Saved notebook outputs
were cleared to remove obsolete plots/metrics; code was executed separately, and
results are recorded here. README comparison scores now match the rebuilt models.
`test_functional.py` adds five functional regression checks using real artifacts.

## Test results

All five functional tests passed in a fresh process. Tested using Windows and
Python 3.12.14, with all eight direct dependencies exactly matching the updated
requirements (including Streamlit 1.45.1 and XGBoost 3.3.0). Existing local package
directories supplied the dependencies; a clean pip installation was not performed.

- Every preprocessing and training code cell ran successfully; all five candidate
  models trained, and the selected models remain XGBoost.
- Regenerated data: 40,575 rows, 28 columns, 24 model features; schema/order match
  both artifacts. Feature parity checked against 40 sampled processed rows plus
  explicit leap-day/hour/weekday checks. Notebook/export ASTs match exactly.
- Tested all 40,575 feature rows and 4,752 combinations spanning all 11 weather
  categories, holiday on/off, four hours, temperatures -40/15/45°C, rain
  0/10.09686/200 mm, snow 0/0.51/50 mm, and cloud cover 0/100%.
- All regression outputs/probabilities finite; probabilities in [0,1], normalized
  within floating-point tolerance; classifier argmax matches class predictions on
  all tested rows. Repeated predictions match exactly on sampled batches.
- Raw rebuilt regressor still has 6 negative dataset predictions and 126 negative
  weather-grid predictions; serving/evaluation clamp these to zero.
- Independent classifier disagrees with the regressed volume's threshold category
  on 1,359 dataset rows and 930 extreme-grid rows. This is expected for separate
  models, now explicitly explained in the UI; probability labels remain attached
  to the classifier.
- Streamlit AppTest: app loads from another working directory; default prediction,
  severe cold/snow, hot/thunderstorm inputs, and a real negative-output case pass
  without exceptions. Separate real server started on localhost port 8510;
  `/` and `/_stcore/health` both returned HTTP 200.
- Regression on original 8,115-row test split, with clipping: MAE **205.8563**,
  RMSE **328.88**, R² **0.972391**. Classification on its original stratified
  8,115-row test split: accuracy **0.919285**, weighted F1 **0.9191**.

Reproduce after installing requirements, from the project root:

```text
python -m unittest -v test_functional
streamlit run app.py
```

## Remaining limitations and unverified items

- Original evaluation design is retained: full-data preprocessing/terciles and
  candidate selection using the test split contaminate final-score interpretation.
  The numerical impact of these choices was not isolated in this change. A proper
  untouched chronological evaluation is separate work, not claimed by these scores.
- Extreme-weather tests establish execution and internal consistency, not accuracy
  for rare weather or future traffic. Snow/temperature extremes extend beyond
  observed training support. High rain inputs are not capped in the app as they
  are during preprocessing; no prediction-changing defect from that mismatch was
  demonstrated. Accuracy/calibration at those extremes remains unverified.
- Holiday propagation covers dates with a raw holiday marker. Missing marker dates,
  observed-holiday policies and future calendar inference are not established;
  future UI predictions depend on the user's manual holiday selection.
- Duplicate timestamps still retain the notebook's original sorted-first weather
  policy. Conflicting weather readings exist, but their targets agree. No automatic
  aggregation was introduced.
- Python 3.11 (declared in runtime.txt), clean dependency installation, deployment,
  and browser visual layout were not tested. Streamlit interaction was tested with
  its official AppTest harness; the server was additionally checked over HTTP.
- Pickle artifacts remain version-dependent. A portable serialization migration,
  automated calendars, evaluation redesign and production hardening are outside
  this targeted functional fix.

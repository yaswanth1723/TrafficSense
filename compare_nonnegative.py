"""Compare count:poisson with the existing settings; never overwrite shipped models.

Run from the project folder: python compare_nonnegative.py
Selection uses validation only. Existing processed data still has full-data
preprocessing; these are comparative scores, not a pristine forecast evaluation.
"""
import json
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent


def metrics(y, prediction):
    p = np.maximum(0, prediction)
    return dict(mae=float(mean_absolute_error(y, p)),
                rmse=float(np.sqrt(mean_squared_error(y, p))),
                r2=float(r2_score(y, p)), raw_negative_count=int(sum(prediction < 0)))


def main():
    df = pd.read_csv(ROOT / 'data/traffic_processed.csv')
    X = df.drop(columns=['date_time', 'traffic_volume', 'traffic_level', 'traffic_level_encoded'])
    y = df.traffic_volume
    train_pool, test = train_test_split(np.arange(len(df)), test_size=.2, random_state=42)
    train, validation = train_test_split(train_pool, test_size=.2, random_state=42)
    report = dict(split_sizes=dict(train=len(train), validation=len(validation), test=len(test)),
                  selection_rule='Poisson validation MAE <= squared-error MAE and RMSE <= 1.05 * squared-error RMSE; all predictions finite and nonnegative.',
                  caveat='Processed data uses full-data preprocessing and historical test results have already been inspected.',
                  models={})
    for name, objective in [('squared_error', 'reg:squarederror'), ('poisson', 'count:poisson')]:
        model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=.05,
                             subsample=.8, colsample_bytree=.8, random_state=42,
                             n_jobs=-1, objective=objective)
        model.fit(X.iloc[train], y.iloc[train])
        result = {}
        for label, indices in [('validation', validation), ('test', test)]:
            prediction = model.predict(X.iloc[indices])
            assert np.isfinite(prediction).all()
            result[label] = metrics(y.iloc[indices], prediction)
        all_predictions = model.predict(X)
        result['all_rows_finite'] = bool(np.isfinite(all_predictions).all())
        result['all_rows_min'] = float(all_predictions.min())
        # Full Cartesian stress grid using actual processed row schema.
        edge = pd.concat([X.iloc[[0]].astype(float) for _ in range(4752)], ignore_index=True)
        combos = itertools.product([-40, 15, 45], [0, 10.09686, 200], [0, .51, 50],
                                    [0, 100], range(11), [0, 1], [0, 8, 17, 23])
        weather_columns = [c for c in X if c.startswith('weather_')]
        for i, (t, r, s, cloud, weather, holiday, hour) in enumerate(combos):
            edge.loc[i, ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'is_holiday',
                         'month', 'day', 'hour', 'day_of_week', 'is_weekend',
                         'hour_sin', 'hour_cos', 'month_sin', 'month_cos']] = [
                t + 273.15, r, s, cloud, holiday, 12, 25, hour, 4, 0,
                np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                np.sin(2*np.pi*12/12), np.cos(2*np.pi*12/12)]
            edge.loc[i, weather_columns] = [int(weather == j+1) for j in range(10)]
        ep = model.predict(edge)
        result['extreme_rows'] = len(edge)
        result['extreme_finite'] = bool(np.isfinite(ep).all())
        result['extreme_min'] = float(ep.min())
        result['extreme_negatives'] = int(sum(ep < 0))
        report['models'][name] = result
    baseline = report['models']['squared_error']['validation']
    poisson = report['models']['poisson']
    report['poisson_passes_selection'] = bool(
        poisson['validation']['mae'] <= baseline['mae']
        and poisson['validation']['rmse'] <= 1.05 * baseline['rmse']
        and poisson['all_rows_finite'] and poisson['all_rows_min'] >= 0
        and poisson['extreme_finite'] and poisson['extreme_min'] >= 0)
    report['shipped_model_changed'] = False
    (ROOT / 'nonnegative_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

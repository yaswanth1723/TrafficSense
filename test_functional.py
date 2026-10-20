"""Functional checks: python -m unittest -v test_functional (project requirements)."""
import ast
import datetime as dt
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent
# Read the actual feature function without executing Streamlit's top-level UI.
tree = ast.parse((ROOT / 'app.py').read_text(encoding='utf8'))
nodes = [n for n in tree.body if
         (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
          and n.targets[0].id in ['MODEL_FEATURE_ORDER', 'WEATHER_OPTIONS'])
         or (isinstance(n, ast.FunctionDef) and n.name == 'build_feature_row')]
scope = dict(dt=dt, np=np, pd=pd)
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'app_features', 'exec'), scope)
build = scope['build_feature_row']


class FunctionalChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(ROOT / 'data/traffic_processed.csv')
        cls.X = cls.df[scope['MODEL_FEATURE_ORDER']]
        cls.reg = joblib.load(ROOT / 'models/xgb_regressor.pkl')
        cls.clf = joblib.load(ROOT / 'models/xgb_classifier.pkl')

    def test_holiday_dates(self):
        raw = pd.read_csv(ROOT / 'data/Traffic_Volume.csv', parse_dates=['date_time'])
        dates = raw.loc[raw.holiday.fillna('None').ne('None'), 'date_time'].dt.normalize().unique()
        expected = pd.to_datetime(self.df.date_time).dt.normalize().isin(dates).astype(int)
        np.testing.assert_array_equal(expected, self.df.is_holiday)
        self.assertEqual(int(expected.sum()), 1203)

    def test_schema_and_feature_parity(self):
        self.assertEqual(self.reg.get_booster().feature_names, list(self.X))
        self.assertEqual(self.clf.get_booster().feature_names, list(self.X))
        for index in np.linspace(0, len(self.df)-1, 40, dtype=int):
            row = self.df.iloc[index]
            timestamp = pd.Timestamp(row.date_time)
            weather = next((w for w in scope['WEATHER_OPTIONS'][1:] if row['weather_' + w]), 'Clear')
            actual = build(timestamp.date(), timestamp.time(), row.temp - 273.15,
                           row.rain_1h, row.snow_1h, int(row.clouds_all),
                           'Christmas Day' if row.is_holiday else 'None', weather)
            np.testing.assert_allclose(actual.iloc[0].astype(float), self.X.iloc[index].astype(float), atol=1e-12)
        leap = build(dt.date(2024, 2, 29), dt.time(23), 0, 0, 0, 0, 'None', 'Clear').iloc[0]
        self.assertEqual((leap.month, leap.day, leap.hour, leap.day_of_week, leap.is_weekend), (2, 29, 23, 3, 0))

    def test_notebook_exports_and_save_to_missing_models_folder(self):
        for filename in ['01_EDA_and_Preprocessing.ipynb', '02_Model_Training.ipynb']:
            nb = json.loads((ROOT / filename).read_text(encoding='utf8'))
            code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
            export = (ROOT / '.virtual_documents' / filename).read_text(encoding='utf8')
            self.assertEqual(ast.dump(ast.parse(code)), ast.dump(ast.parse(export)))
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                scope = dict(joblib=joblib, best_regressor=self.reg, best_classifier=self.clf)
                for index in [22, 40]:
                    exec(''.join(nb['cells'][index]['source']), scope)
                restored = joblib.load('models/xgb_regressor.pkl')
                np.testing.assert_array_equal(restored.predict(self.X.iloc[:10]), self.reg.predict(self.X.iloc[:10]))
                restored_clf = joblib.load('models/xgb_classifier.pkl')
                np.testing.assert_array_equal(restored_clf.predict_proba(self.X.iloc[:10]), self.clf.predict_proba(self.X.iloc[:10]))
            finally:
                os.chdir(previous)

    def test_all_rows_and_weather_extremes(self):
        combinations = itertools.product([-40, 15, 45], [0, 10.09686, 200], [0, .51, 50],
                                         [0, 100], scope['WEATHER_OPTIONS'],
                                         ['None', 'Christmas Day'], [0, 8, 17, 23])
        edge = pd.concat([build(dt.date(2026, 12, 25), dt.time(h), t, r, s, cloud, hol, w)
                          for t, r, s, cloud, w, hol, h in combinations], ignore_index=True)
        for label, X in [('dataset', self.X), ('extremes', edge)]:
            volume = self.reg.predict(X)
            level = self.clf.predict(X)
            probability = self.clf.predict_proba(X)
            self.assertTrue(np.isfinite(volume).all())
            self.assertTrue(np.isfinite(probability).all())
            self.assertTrue(((probability >= 0) & (probability <= 1)).all())
            np.testing.assert_allclose(probability.sum(axis=1), 1, atol=3e-7)
            np.testing.assert_array_equal(self.clf.classes_[probability.argmax(axis=1)], level)
            np.testing.assert_array_equal(volume[:100], self.reg.predict(X.iloc[:100]))
            served = np.maximum(0, volume)
            disagreement = np.count_nonzero(np.where(served <= 2274, 0, np.where(served <= 4598, 1, 2)) != level)
            print(f'{label}: {len(X)} rows; raw negatives={sum(volume < 0)}; independent disagreements={disagreement}')
        _, test = train_test_split(np.arange(len(self.df)), test_size=.2, random_state=42)
        pred = np.maximum(0, self.reg.predict(self.X.iloc[test]))
        print('Regression MAE / R2:', mean_absolute_error(self.df.traffic_volume.iloc[test], pred),
              r2_score(self.df.traffic_volume.iloc[test], pred))
        self.assertLess(mean_absolute_error(self.df.traffic_volume.iloc[test], pred), 300)
        _, test_c = train_test_split(np.arange(len(self.df)), test_size=.2, random_state=42,
                                    stratify=self.df.traffic_level_encoded)
        score = accuracy_score(self.df.traffic_level_encoded.iloc[test_c], self.clf.predict(self.X.iloc[test_c]))
        self.assertGreater(score, .9)
        print('Classifier accuracy:', score)

    def test_app_normal_extreme_and_negative_predictions_from_other_directory(self):
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as folder:
                os.chdir(folder)
                at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
                self.assertEqual(len(at.exception), 0)
                self.assertEqual(len(at.error), 0)
                at.button[0].click().run()
                self.assertEqual(len(at.exception), 0)
                self.assertEqual(len(at.metric), 1)
                for temp, rain, snow, clouds, weather in [(-40., 200., 50., 100, 'Snow'), (45., 200., 0., 0, 'Thunderstorm')]:
                    at.slider[0].set_value(temp)
                    at.slider[1].set_value(clouds)
                    at.number_input[0].set_value(rain)
                    at.number_input[1].set_value(snow)
                    at.selectbox[0].select('Christmas Day')
                    at.selectbox[1].select(weather)
                    at.button[0].click().run()
                    self.assertEqual(len(at.exception), 0)
                    self.assertFalse(at.metric[0].value.startswith('-'))
                # Exercise an actual negative model prediction through the real UI.
                raw = self.reg.predict(self.X)
                index = np.flatnonzero(raw < 0)[0]
                row = self.df.iloc[index]
                stamp = pd.Timestamp(row.date_time)
                weather = next((w for w in scope['WEATHER_OPTIONS'][1:] if row['weather_' + w]), 'Clear')
                at.date_input[0].set_value(stamp.date())
                at.time_input[0].set_value(stamp.time())
                at.slider[0].set_value(float(row.temp - 273.15))
                at.slider[1].set_value(int(row.clouds_all))
                at.number_input[0].set_value(float(row.rain_1h))
                at.number_input[1].set_value(float(row.snow_1h))
                at.selectbox[0].select('Christmas Day' if row.is_holiday else 'None')
                at.selectbox[1].select(weather)
                at.button[0].click().run()
                self.assertEqual(len(at.exception), 0)
                self.assertEqual(at.metric[0].value, '0 vehicles/hour')
                print(f'UI negative regression case: {stamp}, raw={raw[index]:.2f}, displayed={at.metric[0].value}')
                os.chdir(previous)
        finally:
            os.chdir(previous)


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""User-facing checks against the regressor's fitted weather ranges."""
import math

LABELS = {'temp': ('Temperature', '°C'), 'rain_1h': ('Rain', 'mm'),
          'snow_1h': ('Snow', 'mm'), 'clouds_all': ('Cloud cover', '%')}


def range_warnings(values, ranges):
    messages = []
    for feature, (label, unit) in LABELS.items():
        value = float(values[feature])
        lower, upper = ranges[feature]
        if not math.isfinite(value):
            raise ValueError(f'{label} must be a finite number.')
        if value < lower - 1e-9 or value > upper + 1e-9:
            # Temperature is supplied in Kelvin, as in training; show Celsius.
            offset = 273.15 if feature == 'temp' else 0
            messages.append(
                f'{label} ({value-offset:.6g} {unit}) is outside the regressor training range '
                f'({lower-offset:.6g} to {upper-offset:.6g} {unit}). '
                'This prediction is outside observed training support and may be unreliable.'
            )
    return messages

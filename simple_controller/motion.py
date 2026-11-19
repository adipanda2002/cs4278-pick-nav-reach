"""Rest-to-rest position schedules over a joint-space edge."""
from math import ceil, sqrt

import numpy as np


def smooth_fractions(delta, velocity, acceleration_time, call_duration):
    """Use quintic time scaling with bounded command velocity and acceleration."""
    peak_velocity = 1.875 * np.abs(delta)
    peak_acceleration = (10 / sqrt(3)) * np.abs(delta)
    acceleration = velocity / acceleration_time
    duration = max(np.max(peak_velocity / velocity), np.max(np.sqrt(peak_acceleration / acceleration)))
    calls = max(1, ceil(duration / call_duration))
    u = np.arange(1, calls + 1, dtype=float) / calls
    return u**3 * (10 + u * (-15 + 6 * u))


def cosine_fractions(delta, velocity, acceleration, call_duration):
    """Use half-cosine speed ramps and a possible steady-speed middle."""
    moving = np.abs(delta) > 0.
    if not moving.any():
        return np.array([1.])
    displacement = np.abs(delta[moving])
    rate_limit = np.min(velocity[moving] / displacement)
    acceleration_limit = np.min(acceleration[moving] / displacement)
    peak = min(rate_limit, sqrt(2 * acceleration_limit / np.pi))
    ramp = np.pi * peak / (2 * acceleration_limit)
    cruise = max(0., 1 / peak - ramp)
    duration = 2 * ramp + cruise
    calls = ceil(duration / call_duration)
    elapsed = np.arange(1, calls + 1) * (duration / calls)

    def ramp_distance(t):
        return .5 * peak * (t - ramp / np.pi * np.sin(np.pi * t / ramp))

    fractions = np.where(elapsed < ramp, ramp_distance(elapsed),
                         np.where(elapsed <= ramp + cruise,
                                  .5 * peak * ramp + peak * (elapsed - ramp),
                                  1 - ramp_distance(duration - elapsed)))
    fractions[-1] = 1.
    return fractions


def differential_open_target(nominal, predicted, lower, upper, *, require_near_stop=True):
    """Redistribute opening without changing its sum."""
    high = int(np.argmax(predicted))
    low = 1 - high
    error = .5 * (predicted[high] - predicted[low] - nominal[high] + nominal[low])
    if error < .0015 or (require_near_stop and predicted[high] < upper[high] - .0005):
        return None
    offset = min(error, .004, nominal[high] - lower[high], upper[low] - nominal[low])
    if offset <= 0.:
        return None
    target = nominal.copy()
    target[high] -= offset
    target[low] += offset
    return np.clip(target, lower, upper)

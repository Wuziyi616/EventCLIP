import numpy as np


def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    """Spatially shift events by a random offset."""
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=(2, ))
    events[:, 0] += x_shift
    events[:, 1] += y_shift

    valid_events = (events[:, 0] >= 0) & (events[:, 0] < W) & \
        (events[:, 1] >= 0) & (events[:, 1] < H)
    events = events[valid_events]

    return events


def random_flip_events_along_x(events, resolution=(180, 240), p=0.5):
    """Flip events along horizontally with probability p."""
    H, W = resolution
    if np.random.random() < p:
        events[:, 0] = W - 1 - events[:, 0]
    return events


def random_time_flip_events(events, p=0.5):
    """Flip events over time with probability p."""
    if np.random.random() < p:
        events = np.flip(events, axis=0)
        events = np.ascontiguousarray(events)
        # reverse the time
        events[:, 2] = events[0, 2] - events[:, 2]
        # reverse the polarity
        events[:, 3] = -events[:, 3]
    return events


def center_events(events, resolution=(180, 240)):
    """Center the temporal & spatial coordinates of events.
    Make min_t == 0.
    Make (max_x + min_x + 1) / 2 == W / 2 and (max_y + min_y + 1) / 2 == H / 2.

    Args:
        events: [N, 4 (x,y,t,p)]
        resolution: (H, W)
    """
    # temporal
    events[:, 2] -= events[:, 2].min()
    # spatial
    H, W = resolution
    x_min, x_max = events[:, 0].min(), events[:, 0].max()
    y_min, y_max = events[:, 1].min(), events[:, 1].max()
    x_shift = ((x_max + x_min + 1.) - W) // 2.
    y_shift = ((y_max + y_min + 1.) - H) // 2.
    events[:, 0] -= x_shift
    events[:, 1] -= y_shift
    return events

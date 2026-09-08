import numpy as np
import pandas as pd
from .geometry import smooth_series
from fastf1.logger import get_logger

_logger = get_logger(__name__)

def _compute_telemetry_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes metrics for speed, lateral G, longitudinal G, and elevation gradient."""
    df = df.copy()

    # Map columns safely regardless of case
    col_map = {col.lower(): col for col in df.columns}
    
    speed_col = col_map.get('speed', 'Speed')
    x_col = col_map.get('x', 'X')
    y_col = col_map.get('y', 'Y')
    z_col = col_map.get('z', 'Z')

    # Convert units
    # FastF1 X, Y, Z coordinates are in decimeters -> convert to meters (/ 10.0)
    # Speed is in km/h -> convert to m/s (/ 3.6)
    speed_kmh = df[speed_col].astype(float)
    v_ms = speed_kmh / 3.6
    
    x_m = df[x_col].astype(float) / 10.0
    y_m = df[y_col].astype(float) / 10.0
    z_m = df[z_col].astype(float) / 10.0 if z_col in df.columns else np.zeros(len(df))

    # time delta (dt in seconds)
    if 'Time' in df.columns and pd.api.types.is_timedelta64_dtype(df['Time']):
        t_sec = df['Time'].dt.total_seconds().to_numpy()
    elif 'Date' in df.columns:
        t_sec = df['Date'].diff().dt.total_seconds().fillna(0.1).cumsum().to_numpy()
    else:
        print("Use Fallback: assume 10Hz")
        t_sec = np.arange(len(df)) * 0.1  

    dt = np.gradient(t_sec)
    dt = np.where(dt <= 0.001, 0.1, dt)  # Prevent division by microscopic dt steps

    # 4. Smooth coordinates and speed to filter out GPS positioning jitter
    x_smooth = _smooth_series(x_m, window=21, polyorder=2)
    y_smooth = _smooth_series(y_m, window=21, polyorder=2)
    z_smooth = _smooth_series(z_m, window=21, polyorder=2)
    v_smooth = _smooth_series(v_ms, window=15, polyorder=2)

    # 5. Longitudinal G-Force: a_lon = (1/g) * (dv / dt)
    dv_dt = np.gradient(v_smooth, t_sec)
    lon_g = dv_dt / 9.81

    # 6. Lateral G-Force: a_lat = (1/g) * v * |d_heading / dt|
    dx = np.gradient(x_smooth, t_sec)
    dy = np.gradient(y_smooth, t_sec)
    heading = np.unwrap(np.arctan2(dy, dx))
    heading_smooth = _smooth_series(pd.Series(heading), window=15, polyorder=2)
    dheading_dt = np.gradient(heading_smooth, t_sec)
    
    lat_g = (v_smooth * np.abs(dheading_dt)) / 9.81
    # Zero out lateral G at low speeds (< 30 km/h) where heading flips randomly
    lat_g = np.where(speed_kmh < 30.0, 0.0, lat_g)

    # 7. Elevation Gradient (%): (dz / d_distance) * 100
    dz = np.gradient(z_smooth)
    dist_step = np.sqrt(np.gradient(x_smooth)**2 + np.gradient(y_smooth)**2)
    gradient_pct = np.where(dist_step > 0.05, (dz / dist_step) * 100.0, 0.0)

    # Assign calculated metrics
    df['speed'] = speed_kmh
    df['elevation'] = gradient_pct
    df['lat_g'] = np.clip(lat_g, 0.0, 6.5)
    df['lon_g'] = np.clip(lon_g, -6.5, 6.5)
    return df

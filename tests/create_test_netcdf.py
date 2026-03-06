"""Helper script to create the test NetCDF fixture.

Run once: python tests/create_test_netcdf.py
"""

import numpy as np

# Use netCDF4 directly to avoid xarray dep issues in CI
try:
    from netCDF4 import Dataset
except ImportError:
    # Fallback: create a minimal file with xarray
    import xarray as xr
    import pandas as pd

    times = pd.date_range("2024-10-09", periods=24, freq="h")
    lats = np.array([39.0, 39.5, 40.0, 40.5, 41.0])
    lons = np.array([-75.0, -74.5, -74.0, -73.5, -73.0])

    np.random.seed(42)
    zeta = np.random.uniform(-0.5, 1.5, size=(24, 5, 5)).astype(np.float32)
    temp = np.random.uniform(10.0, 25.0, size=(24, 5, 5)).astype(np.float32)

    ds = xr.Dataset(
        {
            "zeta": (["time", "lat", "lon"], zeta, {"units": "m", "long_name": "sea surface elevation"}),
            "temp": (["time", "lat", "lon"], temp, {"units": "degC", "long_name": "sea surface temperature"}),
        },
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={"title": "Test STOFS output", "model": "STOFS-2D-Global"},
    )
    ds.to_netcdf("tests/fixtures/sample_netcdf.nc")
    print("Created tests/fixtures/sample_netcdf.nc via xarray")
    raise SystemExit(0)


from pathlib import Path

out_path = Path(__file__).parent / "fixtures" / "sample_netcdf.nc"

nc = Dataset(str(out_path), "w", format="NETCDF4")

# Dimensions
nc.createDimension("time", 24)
nc.createDimension("lat", 5)
nc.createDimension("lon", 5)

# Coordinates
time_var = nc.createVariable("time", "f8", ("time",))
time_var.units = "hours since 2024-10-09 00:00:00"
time_var.calendar = "standard"
time_var[:] = np.arange(24)

lat_var = nc.createVariable("lat", "f4", ("lat",))
lat_var.units = "degrees_north"
lat_var[:] = np.array([39.0, 39.5, 40.0, 40.5, 41.0])

lon_var = nc.createVariable("lon", "f4", ("lon",))
lon_var.units = "degrees_east"
lon_var[:] = np.array([-75.0, -74.5, -74.0, -73.5, -73.0])

# Data variables
np.random.seed(42)
zeta = nc.createVariable("zeta", "f4", ("time", "lat", "lon"))
zeta.units = "m"
zeta.long_name = "sea surface elevation"
zeta[:] = np.random.uniform(-0.5, 1.5, size=(24, 5, 5))

temp = nc.createVariable("temp", "f4", ("time", "lat", "lon"))
temp.units = "degC"
temp.long_name = "sea surface temperature"
temp[:] = np.random.uniform(10.0, 25.0, size=(24, 5, 5))

# Global attributes
nc.title = "Test STOFS output"
nc.model = "STOFS-2D-Global"

nc.close()
print(f"Created {out_path}")

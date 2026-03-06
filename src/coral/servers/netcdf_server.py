"""MCP server for reading and querying NetCDF files on the local filesystem."""

from __future__ import annotations

import json

import numpy as np
import xarray as xr
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("coral-netcdf")


@mcp.tool()
def inspect_netcdf(file_path: str) -> str:
    """Inspect a NetCDF file: show dimensions, variables, global attributes.

    Args:
        file_path: Path to the NetCDF file on the filesystem.
    """
    ds = xr.open_dataset(file_path)
    info = {
        "dimensions": dict(ds.sizes),
        "variables": {},
        "global_attrs": {k: str(v) for k, v in ds.attrs.items()},
    }
    for var_name, var in ds.data_vars.items():
        info["variables"][var_name] = {
            "dims": list(var.dims),
            "shape": list(var.shape),
            "dtype": str(var.dtype),
            "units": var.attrs.get("units", ""),
            "long_name": var.attrs.get("long_name", ""),
        }
    ds.close()
    return json.dumps(info, indent=2, default=str)


@mcp.tool()
def query_netcdf(
    file_path: str,
    variable: str,
    lat: float = None,
    lon: float = None,
    time: str = None,
    time_index: int = None,
) -> str:
    """Query a specific variable from a NetCDF file at a given location and/or time.

    Args:
        file_path: Path to the NetCDF file.
        variable: Variable name to query (e.g., 'zeta', 'temp', 'elev').
        lat: Latitude for nearest-neighbor selection.
        lon: Longitude for nearest-neighbor selection.
        time: Time string for selection (e.g., '2024-10-09T12:00:00').
        time_index: Integer time index (alternative to time string).
    """
    ds = xr.open_dataset(file_path)
    da = ds[variable]

    # Time selection
    if time is not None:
        da = da.sel(time=time, method="nearest")
    elif time_index is not None:
        da = da.isel(time=time_index)

    # Spatial selection — try common coordinate names
    sel = {}
    if lat is not None:
        for coord in ("lat", "latitude", "y"):
            if coord in da.dims:
                sel[coord] = lat
                break
    if lon is not None:
        for coord in ("lon", "longitude", "x"):
            if coord in da.dims:
                sel[coord] = lon
                break
    if sel:
        da = da.sel(**sel, method="nearest")

    values = da.values
    units = da.attrs.get("units", "")

    if values.size == 1:
        result = f"{variable} = {float(values):.4f} {units}"
    elif values.size <= 20:
        result = f"{variable} = {values.tolist()} {units}"
    else:
        result = (
            f"{variable}: shape={values.shape}, "
            f"min={float(np.nanmin(values)):.4f}, "
            f"max={float(np.nanmax(values)):.4f}, "
            f"mean={float(np.nanmean(values)):.4f} {units}"
        )

    ds.close()
    return result


@mcp.tool()
def get_netcdf_timeseries(
    file_path: str, variable: str, lat: float, lon: float
) -> str:
    """Extract a full time series of a variable at a given lat/lon from a NetCDF file.

    Returns JSON array of {time, value} pairs.

    Args:
        file_path: Path to the NetCDF file.
        variable: Variable name.
        lat: Latitude.
        lon: Longitude.
    """
    ds = xr.open_dataset(file_path)
    da = ds[variable]

    # Find spatial coordinates
    sel = {}
    for coord in ("lat", "latitude", "y"):
        if coord in da.dims:
            sel[coord] = lat
            break
    for coord in ("lon", "longitude", "x"):
        if coord in da.dims:
            sel[coord] = lon
            break

    if sel:
        da = da.sel(**sel, method="nearest")

    records = []
    if "time" in da.dims:
        for t in da.time.values:
            val = float(da.sel(time=t).values)
            records.append({"time": str(t), "value": round(val, 6)})
    else:
        records.append({"value": float(da.values)})

    ds.close()
    return json.dumps(records[:500])


if __name__ == "__main__":
    mcp.run()

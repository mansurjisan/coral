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
def get_netcdf_timeseries(file_path: str, variable: str, lat: float, lon: float) -> str:
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


@mcp.tool()
def netcdf_stats(
    file_path: str,
    variable: str,
    time_start: str = None,
    time_end: str = None,
    stat: str = "all",
) -> str:
    """Compute statistics for a NetCDF variable over a time range.

    Supports natural-language style queries like "What was the max water level
    between March 10 and March 15?" by specifying variable, time range, and stat.

    Args:
        file_path: Path to the NetCDF file.
        variable: Variable name (e.g. 'zeta', 'temp', 'salt', 'elev').
        time_start: Start time for the range (e.g. '2026-03-10'). Optional.
        time_end: End time for the range (e.g. '2026-03-15'). Optional.
        stat: Statistic to compute: 'min', 'max', 'mean', 'std', 'all'.
            Default 'all' returns all stats.
    """
    ds = xr.open_dataset(file_path)
    if variable not in ds.data_vars:
        available = list(ds.data_vars)
        ds.close()
        return f"Variable '{variable}' not found. Available: {', '.join(available)}"

    da = ds[variable]

    # Time slicing
    if "time" in da.dims:
        if time_start and time_end:
            da = da.sel(time=slice(time_start, time_end))
        elif time_start:
            da = da.sel(time=slice(time_start, None))
        elif time_end:
            da = da.sel(time=slice(None, time_end))

    units = da.attrs.get("units", "")
    values = da.values

    n_times = da.sizes.get("time", 1) if "time" in da.dims else 1
    results = {
        "variable": variable,
        "units": units,
        "time_steps": n_times,
    }

    if stat in ("min", "all"):
        results["min"] = round(float(np.nanmin(values)), 6)
    if stat in ("max", "all"):
        results["max"] = round(float(np.nanmax(values)), 6)
    if stat in ("mean", "all"):
        results["mean"] = round(float(np.nanmean(values)), 6)
    if stat in ("std", "all"):
        results["std"] = round(float(np.nanstd(values)), 6)

    nan_count = int(np.isnan(values).sum()) if np.issubdtype(values.dtype, np.floating) else 0
    results["nan_count"] = nan_count
    results["total_values"] = int(values.size)

    # Time range info
    if "time" in da.dims and n_times > 0:
        results["time_range"] = {
            "start": str(da.time.values[0]),
            "end": str(da.time.values[-1]),
        }

    ds.close()

    # Format as readable output
    lines = [f"## {variable} Statistics"]
    if results.get("time_range"):
        lines.append(f"**Time range**: {results['time_range']['start']} to {results['time_range']['end']}")
    lines.append(f"**Time steps**: {results['time_steps']}")
    if "min" in results:
        lines.append(f"**Min**: {results['min']} {units}")
    if "max" in results:
        lines.append(f"**Max**: {results['max']} {units}")
    if "mean" in results:
        lines.append(f"**Mean**: {results['mean']} {units}")
    if "std" in results:
        lines.append(f"**Std**: {results['std']} {units}")
    if nan_count > 0:
        lines.append(f"**NaN values**: {nan_count} / {results['total_values']}")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()

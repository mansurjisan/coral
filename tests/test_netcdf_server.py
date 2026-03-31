"""Tests for the NetCDF MCP server tool functions.

Tests call the tool functions directly (not via MCP protocol).
"""

import json
from pathlib import Path

import pytest

from coral.servers.netcdf_server import get_netcdf_timeseries, inspect_netcdf, query_netcdf

FIXTURES = Path(__file__).parent / "fixtures"
NC_PATH = str(FIXTURES / "sample_netcdf.nc")


class TestInspectNetcdf:
    def test_returns_valid_json(self):
        result = inspect_netcdf(NC_PATH)
        data = json.loads(result)
        assert "dimensions" in data
        assert "variables" in data
        assert "global_attrs" in data

    def test_dimensions(self):
        data = json.loads(inspect_netcdf(NC_PATH))
        dims = data["dimensions"]
        assert dims["time"] == 24
        assert dims["lat"] == 5
        assert dims["lon"] == 5

    def test_variables_listed(self):
        data = json.loads(inspect_netcdf(NC_PATH))
        assert "zeta" in data["variables"]
        assert "temp" in data["variables"]

    def test_variable_metadata(self):
        data = json.loads(inspect_netcdf(NC_PATH))
        zeta = data["variables"]["zeta"]
        assert zeta["units"] == "m"
        assert zeta["long_name"] == "sea surface elevation"
        assert zeta["shape"] == [24, 5, 5]
        assert zeta["dims"] == ["time", "lat", "lon"]

    def test_global_attrs(self):
        data = json.loads(inspect_netcdf(NC_PATH))
        assert data["global_attrs"]["model"] == "STOFS-2D-Global"

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            inspect_netcdf("/nonexistent/file.nc")


class TestQueryNetcdf:
    def test_single_point(self):
        result = query_netcdf(NC_PATH, "zeta", lat=40.0, lon=-74.0, time_index=0)
        assert "zeta" in result
        assert "m" in result

    def test_time_index(self):
        result = query_netcdf(NC_PATH, "temp", time_index=12)
        # Returns summary of a 5x5 spatial slice
        assert "temp" in result
        assert "degC" in result

    def test_spatial_summary(self):
        result = query_netcdf(NC_PATH, "zeta")
        # Full 24x5x5 array — should return summary stats
        assert "min=" in result
        assert "max=" in result
        assert "mean=" in result

    def test_missing_variable_raises(self):
        with pytest.raises(Exception):
            query_netcdf(NC_PATH, "nonexistent_var")


class TestGetNetcdfTimeseries:
    def test_returns_json_array(self):
        result = get_netcdf_timeseries(NC_PATH, "zeta", lat=40.0, lon=-74.0)
        records = json.loads(result)
        assert isinstance(records, list)
        assert len(records) == 24

    def test_record_structure(self):
        records = json.loads(get_netcdf_timeseries(NC_PATH, "zeta", lat=40.0, lon=-74.0))
        for r in records:
            assert "time" in r
            assert "value" in r
            assert isinstance(r["value"], float)

    def test_different_variable(self):
        records = json.loads(get_netcdf_timeseries(NC_PATH, "temp", lat=39.0, lon=-75.0))
        assert len(records) == 24
        # Temp values should be in reasonable range
        for r in records:
            assert 5.0 < r["value"] < 30.0

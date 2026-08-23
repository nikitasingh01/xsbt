"""Reporting. Takes a finished run and produces something a PM can open and read."""

from xsbt.report.html import (
    ReportData,
    analyse,
    render,
    write_html,
    write_metrics,
    write_returns,
)

__all__ = [
    "ReportData",
    "analyse",
    "render",
    "write_html",
    "write_metrics",
    "write_returns",
]

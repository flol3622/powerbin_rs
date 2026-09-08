"""
PowerBin (Rust): High-Performance Adaptive 2D Data Binning with Centroidal Power Diagrams
"""

from .powerbin import PowerBin, power_diagram, update_bins

__version__ = "0.1.0"
__author__ = "Antigravity Team (Algorithm by Michele Cappellari)"
__all__ = ["PowerBin", "power_diagram", "update_bins"]

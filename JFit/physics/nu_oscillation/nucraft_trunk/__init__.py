#! /usr/bin/env python
"""
The nuCraft package for calculation of atmospheric neutrino oscillation probabilities
consists of the NuCraft module with the main class NuCraft and the auxiliary class
EarthModel, which are both imported to the package namespace.
For more information, see the module's and classes' docstrings.
"""
__all__ = ["NuCraft","EarthModel"]

from NuCraft import EarthModel
from NuCraft import NuCraft

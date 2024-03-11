"""Rapidly generate periodic accretion and flux timeseries templates for eccentric binaries"""

import sys
if not '-m' in sys.argv:
	from .accretion import AccretionSeries
	from .flux import BinaryAlphaDisk


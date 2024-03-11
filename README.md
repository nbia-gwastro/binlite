## binlite  
`binlite` is a Python package for rapidly generating accretion templates and inferred flux timeseries from eccentric binaries   
  
  
### Modules 

`accretion` : generate periodic templates of accretion onto the 'primary', 'secondary', or 'total' of an eccentric binary  
`flux` : calculate periodic flux templates associated with a periodic accretion template  

### Intended Use   

Users can generate a periodic accretion template in two separate ways:  
  
1. Generate an instance of an `AccretionSeries` object and access the data via the associated `time`, `primary`, `secondary`, and `total` attributes; e.g.  
  
  	```python
	import binlite as blt  
	ts = blt.AccretionSeries(0.3, n_orbits=5, n_modes=29)  
	orbits = ts.time  
	mdot_primary = ts.primary  
	mdot_secondary = ts.secondary  
	mdot_total = ts.total  
	```  
  
2. Directly call the functions `accretion.primary`, `accretion.secondary`, `accretion.total`, `accretion.orbits` with the same signature as AccretionSeries objects (will generate an AccretionSeries object as an intermediary); e.g.  
  
	```python
   	from binlite.accretion import orbits, primary, secondary, total  
   	orbits = orbits(0.5, n_orbits=5, n_modes=29, retrograde=True)  
	mdot_primary = primary(0.5, n_orbits=5, n_modes=29, retrograde=True)  
	mdot_secondary = secondary(0.5, n_orbits=5, n_modes=29, retrograde=True)  
	mdot_total = total(0.5, n_orbits=5, n_modes=29, retrograde=True)  
	```

Similarly, users have acces to two means of creating an associated flux timeseries, all of which first require the creation of an `AccretionSeries` object off which to compute the flux variability:

1. Directly call the functions `periodic_flux_series` and `normalized_flux_series` for a flux series or normalized against the total average flux, respectively. These functions require the user to supply the observing frequency in hertz, an `AccretionSeries` object, the orbital period of the binary in years, the total mass of the binary in solar-masses, and the luminosity distance to the binary in parsec. They also optionally accept specifications for the accretion eddington fraction, accretion efficiency, observer inclination angle in degrees, as well as the inner and outer edges of both the minidisks and the outer-disk for the integration fothe blackbody spectra; e.g.

	```python
	from binlite import AccretionSeries
	from binlite.flux import periodic_flux_series, normalized_flux_series
	acc = AccretionSeries(0.4)
	fnu_series = periodic_flux_series(frequency, acc, period_yr, total_mass_msun, luminosity_distance_pc, eddington_ratio=0.1)
	fnu_normal = normalized_flux_series(frequency, acc, period_yr, total_mass_msun, luminosity_distance_pc, eddington_ratio=0.1)
	```  

   The `time` routine is for plotting and generates an array of observation times in years from the associated `AccretionSeries` object and the binary orbital period
  
	```python
	import binlite as blt
	time = blt.flux.time(accretion_series, period_yr)
	```  
  
2. The above routines, as an intermediary, generate an instance of a `BinaryAlphaDisk` object which contains all the information about the system eccentricity, period, mass, distance, accretion rate, efficiency and disk sizes. If preferred, one can themselves generate a `BinaryAlphaDisk` object and calculate flux series from this and an associated `AccretionSeries` object via the `periodic_flux_series_from_bad`, `normalized_flux_series_from_bad`, and `time_from_bad` functions; e.g.

	```python
	import binlite as blt
	acc = blt.AccretionSeries(eccentricity)
	bad = blt.BinaryAlphaDisk(acc.ecc, period_yr, total_mass_msun, luminosity_distance_pc, accretion_efficiency=0.01)
	time = blt.flux.time_from_bad(acc, bad)
	fnu_series = blt.flux.periodic_flux_series_from_bad(frequency, acc, bad)
	fnu_normal = blt.flux.normalized_flux_series_from_bad(frequency, acc, bad)
	```

The flux module also contains a convenience function `magnitude_from_flux` which converts a flux timeseries into a timeseries of apparent magnitudes. The function takes as input any flux series at some frequency and the associated zero-point flux for that observing band: `blt.flux.magnitude_from_flux(fnu_series, zero_point_flux)`

The calculations for generating flux timeseries assume that the outer-disk and each minidisk are described by independent Shakura-Sunyaev alpha-disk solutions. See the associated paper D'Orazio, Duffell & Tiede (2024) for more detail.

Each module can also be run directly by running `python -m binlite.accretion` or `python -m binlite.flux`. For each module this will run a small example script (that additionally requires a `matplotlib` installation; not specified as a dependency). The `accretion` example will generate figures like Figs. 2, 3 & 4 in D'Orazio, Duffell & Tiede (2024), and the `flux` example will create figures like Fig. 7 (without the lensed or boosted components).


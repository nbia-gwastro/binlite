import numpy as np
from scipy.integrate import quad
from .constants import *
from .accretion import AccretionSeries, Qfit


# ===================================================================
class BinaryAlphaDisk:
	"""Helper class for generating periodic flux curves from accretion onto equal-mass eccentric binaries

	Parameters
	----------
	eccentricity: 
		desired binary eccentricity for accretion series (max 0.8)
	period_yr:
		binary's orbital period in years 
	total_mass_msun:
		total binary mass in solar-masses
	luminosity_distance_pc:
		luminosity distance to the source binary in parsec
	eddington_ratio (optional, default=1.0):
		ratio of the total rate onto the binary to the eddington rate
	accretion_efficiency (optional, default=0.1):
		efficiency at which gravitational energy is converted to radiation
	md_inner_edge_risco (optional, default=1.0):
		inner edge of the primary component's minidisk in units of the isco-radius
		for integrating the minidisk spectrum
	cbd_inner_edge_a (optional, default=2.0):
		inner edge of the circumbinary disk in units of the binary semi-major axis distance
		for integrating the outer-disk spectrum
	cbd_outer_edge_a (optional, default=100.0):
		outer edge of the circumbinary disk in units of the binary semi-major axis distance
		for integrating the outer-disk spectrum
	inclination_deg (optional, default=0.0):
		viewing inclination for the coplanar binary-disk system in degrees

	Public methods
	--------------
	fnu_primary   : specific flux from the primary's minidisk at a given frequency
	fnu_secondary : specific flux from the secondary's minidisk at a given frequency
	fnu_disk      : specific flux from the outer-disk at a given frequency
	fnu_total     : specific flux of the full system at a given frequency
	primary_flux_ratio   : specific flux ratio of the primary's minidisk component to the total
	secondary_flux_ratio : specific flux ratio of the secondary's minidisk component to the total
	disk_flux_ratio      : specific flux ratio of the outer-disk component to the total
		
	"""
	def __init__(self, 
				 eccentricity:float,
				 period_yr:float, 
				 total_mass_msun:float, 
				 luminosity_distance_pc:float,
				 eddington_ratio:float=1.0, 
				 accretion_efficiency:float=0.1,
				 md_inner_edge_risco:float=1.0,
				 cbd_inner_edge_a:float=2.0,
				 cbd_outer_edge_a:float=100.0,
				 inclination_deg:float=0.0,  
				 ):
		self.q = 1.0
		self.p = period_yr * yr2sec
		self.m = total_mass_msun * Msun_cgs
		self.i = inclination_deg * (np.pi / 180.)
		self.a = self.__semi_major_axis(self.p, self.m)
		self.mdot = eddington_ratio * self.__eddington_accretion_rate(self.m, accretion_efficiency)
		self.qfac = Qfit(eccentricity)
		self.dlum = luminosity_distance_pc * pc2cm
		self.m1  = self.m    / (1.0 + self.q)
		self.dm1 = self.mdot / (1.0 + self.qfac)
		self.rin_md   = md_inner_edge_risco * self.__risco(self.m1)
		self.rout_md  = self.__truncation_radius_md()
		self.rin_cbd  = cbd_inner_edge_a * self.a
		self.rout_cbd = cbd_outer_edge_a * self.a
		self.temp1 = self.__disk_temperature_r(self.m1, self.dm1, self.rin_md)

	# -------------------------------------------------------------------------
	def fnu_primary(self, nu):
		"""specific flux from the primary's minidisk at a given frequency

		nu : observing frequency in Hz
		"""
		return self.__fnu_disk(nu, self.temp1, self.rin_md, self.rin_md, self.rout_md, self.i, self.dlum)

	def fnu_secondary(self, nu):
		"""specific flux from the secondary's minidisk at a given frequency

		nu : observing frequency in Hz
		"""
		return self.__fnu_disk(nu, self.qfac**(1./4.) * self.temp1, self.rin_md, self.rin_md, self.rout_md, self.i, self.dlum)

	def fnu_disk(self, nu):
		"""specific flux from the outer-disk at a given frequency

		nu : observing frequency in Hz
		"""
		t_fac_cbd = 2.0**(1./4.) * (1.0 + self.qfac)**(1./4.)
		return self.__fnu_disk(nu, t_fac_cbd * self.temp1, self.rin_md, self.rin_cbd, self.rout_cbd, self.i, self.dlum)

	def fnu_total(self, nu):
		"""specific flux of the full system at a given frequency

		nu : observing frequency in Hz
		"""
		return self.fnu_primary(nu) + self.fnu_secondary(nu) + self.fnu_disk(nu)

	def primary_flux_ratio(self, nu):
		"""specific flux ratio of the primary's minidisk component to the total

		nu : observing frequency in Hz
		"""
		return self.fnu_primary(nu) / self.fnu_total(nu)

	def secondary_flux_ratio(self, nu):
		"""specific flux ratio of the secondary's minidisk component to the total

		nu : observing frequency in Hz
		"""
		return self.fnu_secondary(nu) / self.fnu_total(nu)

	def disk_flux_ratio(self, nu):
		"""specific flux ratio of the outer-disk component to the total

		nu : observing frequency in Hz
		"""
		return 1.0 - self.primary_flux_ratio(nu) - self.secondary_flux_ratio(nu)

	# Internal methods
	# -------------------------------------------------------------------------
	def __eddington_luminosity(self, m):
		LEddMsun = 4. * np.pi * G_cgs * Msun_cgs * mp_cgs * c_cgs / sigT_cgs
		return LEddMsun * (m / Msun_cgs)

	def __eddington_accretion_rate(self, m, eta):
		return self.__eddington_luminosity(m) / (eta * c_cgs * c_cgs)  

	def __risco(self, m):
		return 6.0 * G_cgs * m / c_cgs**2

	def __semi_major_axis(self, p, m):
		return (p / (2. * np.pi))**(2./3.) * (G_cgs * m)**(1./3.)

	def __truncation_radius_md(self):
		return 0.27 * self.q**(0.3) * self.a

	def __disk_temperature_r(self, m, dm, r):
		return (3. * G_cgs * m * dm / (8. * np.pi * r**3 * sigSB_cgs))**(1/4)

	def __bnu(self, nu, r, temp_in, r_in):
		exarg = hp_cgs * nu / (kb_cgs * temp_in * (r/r_in)**(-3./4.))
		return 2. * hp_cgs * nu**3 / c_cgs**2 / (np.exp(exarg) - 1.0)

	def __bnu_disk(self, r, nueval, temp_in, r_in):
		return r * self.__bnu(nueval, r, temp_in, r_in)

	def __fnu_disk(self, nu, tpr_min, rpr_min, r_in, r_out, inc, dst):
		prefac = 2.0 * np.pi * np.cos(inc) / dst**2
		return prefac * quad(self.__bnu_disk, r_in, r_out, args=(nu, tpr_min, rpr_min))[0]


# User callable functions: public API
# =============================================================================
def time(accretion_series:AccretionSeries, period_yr:float):
	"""Generate time array for an associated periodic flux series

	Parameters
	----------
	accretion_series:
		an AccretionSeries object (see accretion.py) for binary of supplied eccentricity
		 - holds desired number of orbits
	period_yr:
		binary's orbital period in years 

	Return
	------
	(ndarry) array of observational time in years with same shape as accretion_series.primary/secondary/total
	"""
	norb = accretion_series.orbits
	yrs  = norb * period_yr
	nx   = len(accretion_series.primary)
	return np.linspace(0., yrs, nx)

def normalized_flux_series(frequency:float, 
				           accretion_series:AccretionSeries, 
			               period_yr:float, 
			               total_mass_msun:float, 
			               luminosity_distance_pc:float,  
			               eddington_ratio:float=1.0, 
			               accretion_efficiency:float=0.1,
			               md_inner_edge_risco:float=1.0,
			               cbd_inner_edge_a:float=2.0,
			               cbd_outer_edge_a:float=100.0,
			               inclination:float=0.0,
			              ):
	"""Generate a periodic flux timeseries at given frequency normalized to the total averaged (in the rest frame) flux

	Parameters
	----------
	frequency: 
		observation frequncey in Hz
	accretion_series:
		an AccretionSeries object (see accretion.py) for binary of supplied eccentricity
		 - also holds desired number of orbits and fourier modes in the periodic reconstruction
	period_yr:
		binary's orbital period in years 
	total_mass_msun:
		total binary mass in solar-masses
	luminosity_distance_pc:
		luminosity distance to the source binary in parsec
	eddington_ratio (optional, default=1.0):
		ratio of the total rate onto the binary to the eddington rate
	accretion_efficiency (optional, default=0.1):
		efficiency at which gravitational energy is converted to radiation
	md_inner_edge_risco (optional, default=1.0):
		inner edge of the primary component's minidisk in units of the isco-radius
		for integrating the minidisk spectrum
	cbd_inner_edge_a (optional, default=2.0):
		inner edge of the circumbinary disk in units of the binary semi-major axis distance
		for integrating the outer-disk spectrum
	cbd_outer_edge_a (optional, default=100.0):
		outer edge of the circumbinary disk in units of the binary semi-major axis distance
		for integrating the outer-disk spectrum
	inclination_deg (optional, default=0.0):
		viewing inclination for the coplanar binary-disk system in degrees

	Return
	------
	(ndarry) normalized periodic flux timeseries with same shape as accretion_series.primary/secondary/total in cgs
	"""
	acc = accretion_series
	bad = BinaryAlphaDisk(acc.ecc, period, total_mass_msun, luminosity_distance_pc, eddington_ratio, accretion_efficiency, md_inner_edge_risco, cbd_inner_edge_a, cbd_outer_edge_a, inclination)
	chi1 = bad.primary_flux_ratio(frequency)
	chi2 = bad.secondary_flux_ratio(frequency)
	disk_flux = 1.0 - chi1 - ch2
	mdot_mean = np.mean(acc.total)
	return chi1 * acc.primary / mdot_mean + chi2 * acc.secondary / mdot_mean + disk_flux

def periodic_flux_series(frequency:float, 
				 		 accretion_series:AccretionSeries, 
			     		 period_yr:float, 
			     		 total_mass_msun:float, 
			     		 luminosity_distance_pc:float,
			     		 eddington_ratio:float=1.0, 
			     		 accretion_efficiency:float=0.1,
			     		 md_inner_edge_risco:float=1.0,
			     		 cbd_inner_edge_a:float=2.0,
			     		 cbd_outer_edge_a:float=100.0,
			     		 inclination:float=0.0,
			     		):
	"""Generate a periodic flux timeseries at given frequency

	Parameters
	----------
	frequency: 
		observation frequncey in Hz
	accretion_series:
		an AccretionSeries object (see accretion.py) for binary of supplied eccentricity 
		- also holds desired number of orbits and fourier modes in the periodic reconstruction
	period_yr:
		binary's orbital period in years 
	total_mass_msun:
		total binary mass in solar-masses
	luminosity_distance_pc:
		luminosity distance to the source binary in parsec
	eddington_ratio (optional, default=1.0):
		ratio of the total rate onto the binary to the eddington rate
	accretion_efficiency (optional, default=0.1):
		efficiency at which gravitational energy is converted to radiation
	md_inner_edge_risco (optional, default=1.0):
		inner edge of the primary component's minidisk in units of the isco-radius
		for integrating the minidisk spectrum
	cbd_inner_edge_a (optional, default=2.0):
		inner edge of the circumbinary disk in units of the binary semi-major axis distance
		for integrating the outer-disk spectrum
	cbd_outer_edge_a (optional, default=100.0):
		outer edge of the circumbinary disk in units of the binary semi-major axis distance
		for integrating the outer-disk spectrum
	inclination_deg (optional, default=0.0):
		viewing inclination for the coplanar binary-disk system in degrees

	Return
	------
	(ndarry) periodic flux timeseries with same shape as accretion_series.primary/secondary/total in cgs
	"""
	acc = accretion_series
	bad = BinaryAlphaDisk(acc.ecc, period, total_mass_msun, luminosity_distance_pc, eddington_ratio, accretion_efficiency, md_inner_edge_risco, cbd_inner_edge_a, cbd_outer_edge_a, inclination)
	return bad.fnu_total(frequency) * normazlied_flux_series_from_bad(frequency, ts, bad)

# -----------------------------------------------------------------------------
def time_from_bad(accretion_series:AccretionSeries, bad:BinaryAlphaDisk):
	"""Generate time array for an associated periodic flux series from a pre-generated BinaryAlphaDisk object

	Parameters
	----------
	accretion_series:
		an AccretionSeries object (see accretion.py) for binary of supplied eccentricity
		 - holds desired number of orbits
	bad:
		a BinaryAlphaDisk object containing desired system specifics

	Return
	------
	(ndarry) array of observational time in years with same shape as accretion_series.primary/secondary/total
	"""
	return time(accretion_series, bad.p / yr2sec)

def normazlied_flux_series_from_bad(frequency:float, accretion_series:AccretionSeries, bad:BinaryAlphaDisk):
	"""Generate a normalized periodic flux timeseries at given frequency from a BinaryAlphaDisk object

	Parameters
	----------
	frequency: 
		observation frequncey in Hz
	accretion_series:
		an AccretionSeries object (see accretion.py) for binary of supplied eccentricity
		 - holds desired number of orbits
	bad:
		a BinaryAlphaDisk object containing desired system specifics

	Return
	------
	(ndarry) normalized periodic flux timeseries with same shape as accretion_series.primary/secondary/total in cgs
	"""
	acc  = accretion_series
	chi1 = bad.primary_flux_ratio(frequency)
	chi2 = bad.secondary_flux_ratio(frequency)
	disk_flux = 1.0 - chi1 - chi2
	mdot_mean = np.mean(acc.total)
	return chi1 * acc.primary / mdot_mean + chi2 * acc.secondary / mdot_mean + disk_flux

def periodic_flux_series_from_bad(frequency:float, accretion_series:AccretionSeries, bad:BinaryAlphaDisk):
	"""Generate a periodic flux timeseries at given frequency from a BinaryAlphaDisk object

	Parameters
	----------
	frequency: 
		observation frequncey in Hz
	accretion_series:
		an AccretionSeries object (see accretion.py) for binary of supplied eccentricity
		 - holds desired number of orbits
	bad:
		a BinaryAlphaDisk object containing desired system specifics

	Return
	------
	(ndarry) periodic flux timeseries with same shape as accretion_series.primary/secondary/total in cgs
	"""
	return bad.fnu_total(frequency) * normazlied_flux_series_from_bad(frequency, accretion_series, bad)

# -----------------------------------------------------------------------------
def magnitude_from_flux(specific_flux, zero_point_flux):
	"""Apparent magnitude in an observing band given any specific flux F_\nu and the zero-point (normalizing) flux in that band

	Parameters
	----------
	specific_flux: 
		an array of specific flux timeseries (e.g. generated from periodic_flux_series)
	zero_point_flux:
		the zero-point normalizing flux for the observing band associated with the frequency of specific_flux

	Return
	------
	(ndarry) periodic apparent magnitude timeseries with same shape as accretion_series.primary/secondary/total in cgs
	"""
	return -2.5 * np.log10(specific_flux / zero_point_flux)


# To test script by running the module as a script: python -m binlite.flux
#  - Generates plots to compare with Fig. 7 
#    in D'Orazio, Duffell & Tiede (2024)
#  - requires extra matplotlib dependency
# =============================================================================
if __name__ == '__main__':
	p_yr = 1.0
	m_msun = 2 * 1e9
	fedd = 0.1
	dl_pc = 1.5 * 1e9
	import matplotlib.pyplot as plt
	for x in np.arange(0.2, 1.0, 0.2):
		print('ecc : {:.2f}'.format(x))
		fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=[8, 6])
		plt.subplots_adjust(hspace=0.05)
		acc = AccretionSeries(x, n_modes=29, n_orbits=3, retrograde=False)
		bad = BinaryAlphaDisk(x, p_yr, m_msun, dl_pc, eddington_ratio=fedd)
		yrs = time_from_bad(acc, bad)
		flux = periodic_flux_series_from_bad(vband_nu, acc, bad)
		ax1.plot(yrs, flux, 'C0-')
		ax2.plot(yrs, magnitude_from_flux(flux, vband_fnu0), 'k--')
		ax2.set_xlim([0.0, yrs[-1]])
		ax2.set_ylim([15.8, 14.2])
		ax1.set_ylabel(r'$F_v$')
		ax2.set_ylabel(r'$m_v$')
		ax2.set_xlabel('years')
		ax1.set_title(r'e = {:.2f}'.format(acc.ecc))
		plt.show()
		plt.close()

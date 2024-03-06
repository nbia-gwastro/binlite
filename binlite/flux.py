import numpy as np
import scipy as sc
from .constants import
from .accretion import AccretionSeries


# ===================================================================
def Qfit(ecc):
	""" 
	Fitting func for Q(e) from D'Orazio, Duffell, & Tiede 2024, Eq.4 
	"""
	return (1. - (2. - ecc**2 - 2. * ecc**3)*ecc) / (1. + (2. + ecc**2) * ecc)


# ===================================================================
class BinaryAlphaDisk:
	"""
	Helper class for generating flux curves
	"""
	def __init__(self, 
				 period_yr:float, 
				 eccentricity:float,
				 total_mass_msun:float, 
				 eddington_ratio:float, 
				 luminosity_distance_pc:float,
				 accretion_efficiency:float=0.1,
				 md_inner_edge_risco:float=1.0,
				 cbd_inner_edge_a:float=2.0,
				 cbd_outer_edge_a:float=100.0,
				 inclination:float=0.0,
				 ):
		self.q = 1.0
		self.a = self.__semi_major_axis(self.pbin, self.mass)
		self.i = inclination
		self.pbin = period * yr2sec
		self.mass = total_mass_msun * Msun_cgs
		self.mdot = eddington_ratio * self.__eddington_accretion_rate(self.mass, accretion_efficiency)
		self.qfac = Qfit(eccentricity)
		self.dlum = luminosity_distance_pc * pc2cm
		self.m1  = self.mass / (1.0 + self.q)
		self.dm1 = self.mdot / (1.0 + self.qfac)
		self.temp1 = self.__disk_temperature_r(self.m1, self.dm1, self.rin_md)
		self.rin_md = md_inner_edge_risco * self.__risco(self.m1)
		self.rout_md = self.__truncation_radius_md()
		self.rin_cbd = cbd_inner_edge_a * a
		self.rout_cbd = cbd_outer_edge_a * a

	# -------------------------------------------------------------------------
	def fnu_primary(nu):
		return self.__fnu_disk(nu, self.temp1, self.rin_md, self.rin_md, self.rout_md, self.i, self.dlum)

	def fnu_secondary(nu):
		return self.__fnu_disk(nu, self.qfac**(1./4.) * self.temp1, self.rin_md, self.rin_md, self.rout_md, self.i, self.dlum)

	def fnu_disk(nu):
		t_fac_cbd = 2.0**(1./4.) * (1.0 + self.qfac)**(1./4.)
		return self.__fnu_disk(nu, t_fac_cbd * self.temp1, self.rin_md, self.rin_cbd, self.rout_cbd, self.i, self.dlum)

	def fnu_total(nu):
		return self.fnu_primary(nu) + self.fnu_secondary(nu) + self.fnu_disk(nu)

	def primary_flux_ratio(nu):
		return self.fnu_primary(nu) / self.fnu_total(nu)

	def secondary_flux_ratio(nu):
		return self.fnu_secondary(nu) / self.fnu_total(nu)

	def disk_flux_ratio(nu):
		return 1.0 - self.primary_flux_ratio(nu) - self.secondary_flux_ratio(nu)

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
		return r * __bnu(nueval, r, temp_in, r_in)

	def __fnu_disk(self, nu, tpr_min, rpr_min, r_in, r_out, inc, dst):
		prefac = 2.0 * np.pi * np.cos(inc) / dst**2
		return prefac * sc.integrate.quad(__bnu_disk, r_in, r_out, args=(nu, tpr_min, rpr_min))[0]


# User callable functions
# =============================================================================
def normalized_flux_series(frequency:float, 
				  accretion_series:AccretionSeries, 
			      period:float, 
			      total_mass_msun:float, 
			      luminosity_distance_pc:float,  #TODO: cosmology.py to generate dl from redshift?
			      eddington_ratio:float=1.0, 
			      accretion_efficiency:float=0.1,
			      md_inner_edge_risco:float=1.0,
			      cbd_inner_edge_a:float=2.0,
			      cbd_outer_edge_a:float=100.0,
			      inclination:float=0.0,
			     ):
	ts  = accretion_series
	cbd = BinaryAlphaDisk(period, 
					       ts.ecc, 
					       total_mass_msun, 
					       eddington_ratio, 
					       luminosity_distance_pc, 
					       accretion_efficiency, 
					       md_inner_edge_risco, 
					       cbd_inner_edge_a, 
					       cbd_outer_edge_a, 
					       inclination)
	chi1 = cbd.primary_flux_ratio(frequency)
	chi2 = cbd.secondary_flux_ratio(frequency)
	disk_flux = 1.0 - chi1 - ch2
	return chi1 * ts.primary + chi2 * ts.secondary + disk_flux

def periodic_flux_series(frequency:float, 
				  accretion_series:AccretionSeries, 
			      period:float, 
			      total_mass_msun:float, 
			      luminosity_distance_pc:float,  #TODO: cosmology.py to generate dl from redshift?
			      eddington_ratio:float=1.0, 
			      accretion_efficiency:float=0.1,
			      md_inner_edge_risco:float=1.0,
			      cbd_inner_edge_a:float=2.0,
			      cbd_outer_edge_a:float=100.0,
			      inclination:float=0.0,
			     ):
	ts  = accretion_series
	cbd = BinaryAlphaDisk(period, 
					       ts.ecc, 
					       total_mass_msun, 
					       eddington_ratio, 
					       luminosity_distance_pc, 
					       accretion_efficiency, 
					       md_inner_edge_risco, 
					       cbd_inner_edge_a, 
					       cbd_outer_edge_a, 
					       inclination)
	return cbd.fnu_total * normazlied_flux_series_from_cbd(frequency, ts, cbd)

def normazlied_flux_series_from_cbd(frequency:float, accretion_series:AccretionSeries, cbd:BinaryAlphaDisk):
	chi1 = cbd.primary_flux_ratio(frequency)
	chi2 = cbd.secondary_flux_ratio(frequency)
	disk_flux = 1.0 - chi1 - ch2
	return chi1 * accretion_series.primary + chi2 * accretion_series.secondary + disk_flux

def periodic_flux_series_from_cbd(frequency:float, accretion_series:AccretionSeries, cbd:BinaryAlphaDisk):
	return cbd.fnu_total * normalized_flux_series(frequency, accretion_series, cbd)


# -----------------------------------------------------------------------------
def magnitude_from_flux(flux, zero_point_flux):
	return -2.5 * np.log10(flux / zero_point_flux)


# if (run_diags):
# 	##DIAGS
# 	Pb = 1.0*yr2sec
# 	Mb = 10**8*Msun_cgs
# 	asep = aorb(Pb,Mb)
# 	qb=1.0 ## this is assumed
# 	ebtst = 0.4

# 	Mprm = Mb/(1.0 + qb)
# 	Msec = qb*Mb/(1.0 + qb)


# 	Qfac = Qfit(ebtst)
# 	etaeff = 0.1

# 	Mdot_CBD = Mdot_Edd(Mb, etaeff)
# 	Mdot_prm = Mdot_CBD/(1.0+Qfac)
# 	dstL = 1.5*10**9*pc2cm
# 	dL_pc = 1.5*10**9
# 	inctst = 0.0  ##0 max, pi.2 = 0

# 	nutst = Vbandnu

# 	rinMDp = RISCO(Mprm)
# 	routMDp = mdtrunc(qb, asep)
# 	rprmin = rinMDp
# 	# rinMDs = RISCO(Msec)
# 	rinCBD = 2.0*asep
# 	routCBD = 100.0*asep


# 	Tprmin = Tdisk(Mprm, Mdot_prm, rinMDp)

# 	# Fnu_disk(nutst, Tprmin, rprmin, rinMDp, routMDp, 1.0, 1.0)


# 	Ftot = Fnu_tot(nutst, Qfac, Tprmin, rprmin, rinMDp, routMDp, rinCBD, routCBD, inctst, dstL)
# 	SSX1 = Xhi1(nutst, Qfac, Tprmin, rprmin, rinMDp, routMDp, rinCBD, routCBD)
# 	SSX2 = Xhi2(nutst, Qfac, Tprmin, rprmin, rinMDp, routMDp, rinCBD, routCBD)


# 	flx_props = get_flux_props(nutst, Pb, Mb, qb, Mdot_CBD, Qfit(ebtst), dL_pc)
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
    barycenter_velocity (optional, default=0.0):
        line-of-sight velocity of the system barycenter in units of c
    argumnet_of_pericenter (optional, default=0.0):
        argument of pericenter for eccentric binary orbit (see D'Orazio, Duffell & Tiede 2024)
    spectral_slope_lnln (optional, default=-1.0):
        dlog(Fnu)/dlog(nu) of the emitting spectrum in the observing band for boosting

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
                 barycenter_velocity_c:float=0.0, 
                 argument_of_pericenter_deg:float=0.0,
                 spectral_slope_lnln:float=-1.0,
                 ):
        self.q = 1.0
        self.p = period_yr * yr2sec
        self.m = total_mass_msun * Msun_cgs
        self.i = inclination_deg * (np.pi / 180.) # NOTE: consistent for boosting
        self.a = self.__semi_major_axis(self.p, self.m)
        self.ecc = eccentricity
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
        self.vbary = barycenter_velocity * c_cgs
        self.pomega = argumnet_of_pericenter_deg * (np.pi / 180.)
        self.alphanu = spectral_slope_lnln

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

    def lensing_boosting_magnification(self, fs):
        """lensing boosting 

        fs : fraction of total light coming from the secondary
                == 0 when all from primary
                == 1 when all from secondary
        """
        # Units: 
        # t: seconds
        # vz: v/c (3e8 m/s)
        # w, Omega, I: rad
        # e, alpha: unitless
        # T, t0: years
        # M_1, M_2: solar masses e8
    
        # Convert units
        sinI = np.sin(self.i)
        t0 = 0.0
        M_1 = self.m / (1. + self.q)
        M_2 = self.q * M_1
            
        # More constants
        Omega = np.pi / 2. # doesn't matter
        n = 2 * np.pi / self.p
        
        a = ((self.p / (2 * np.pi))**2 * G_cgs * (M_1 + M_2))**(1./3.)
        a1 = a * M_2 / (M_1 + M_2)
        a2 = a * M_1 / (M_1 + M_2)
        K1 = (n * a1 * sinI) / (np.sqrt(1. - self.ecc**2))
        K2 = (n * a2 * sinI) / (np.sqrt(1. - self.ecc**2))
        
        ''' Calculate line of sight velocities '''
        # Make arrays
        seconds = t 
        earr = np.full(seconds.size, e)
        Marr = n * (seconds - t0) # mean anomaly
        Earr = eccentric_anomaly(Marr, earr) # Ecc anom : Need to add function somewhere

        rr = np.full(seconds.size, a) * (1. - (self.ecc * np.cos(Earr)))
        #
        r1 = np.full(seconds.size, a1) * (1. - (self.ecc * np.cos(Earr)))
        r2 = np.full(seconds.size, a2) * (1. - (self.ecc * np.cos(Earr)))
                
        arg = np.sqrt((1. + earr) / (1. - earr)) * np.tan(Earr / 2.)
        f = 2. * np.arctan(arg) # true anomaly
              
        vr1 = self.vbary + K1 * (np.cos(self.pomega + f) + earr * np.cos(self.pomega))
        vr2 = self.vbary - K2 * (np.cos(self.pomega + f) + earr * np.cos(self.pomega))
       
        X1 = r1 * ( np.cos(Omega) * np.cos(self.pomega + f) - np.sin(Omega) * np.sin(self.pomega + f) * cosI)
        Y1 = r1 * ( np.sin(Omega) * np.cos(self.pomega + f) + np.cos(Omega) * np.sin(self.pomega + f) * cosI)
        Z1 = r1 * ( np.sin(w + f) * sinI )

        # Remove any points where Z1 = 0
        # TODO : a little janky -- could clean / test
        Z1_i = np.argmin(abs(Z1))
        if Z1[Z1_i] == 0:
            if Z1_i != 0:
                Z1[Z1_i] = Z1[Z1_i - 1]
            else:
                Z1[Z1_i] = Z1[Z1_i + 1]
            
        # m2
        X2 = -1 * X1 * (M_1 / M_2)
        Y2 = -1 * Y1 * (M_1 / M_2)
        Z2 = -1 * Z1 * (M_1 / M_2)
                
        D_s = np.zeros(X1.size)
        D_l = np.zeros(X1.size)
        M_s = np.zeros(X1.size)
        M_l = np.zeros(X1.size)
        magnification = np.zeros(X1.size)

        ## make sure the (v/c)^2 isnt to close to 1 and makes gam a nan [might be very rare]
        #vorb1_sqr = np.minimum(  G_SI*(M_1 + M_2)*Msun_SI *(2./(r1) - 1./a1) /cc/cc,  0.9999999999)
        #vorb2_sqr = np.minimum(  G_SI*(M_1 + M_2)*Msun_SI *(2./(r2) - 1./a2) /cc/cc , 0.9999999999)
        #DD May 2022

        # TODO 0.99999 --> 1. - 1e-10?
        vorb1_sqr = np.minimum( (M_2 / (M_1 + M_2))**2 * G_cgs * (M_1 + M_2) * (2./(rr) - 1./a) / c_cgs / c_cgs, 0.9999999999)
        vorb2_sqr = np.minimum( (M_1 / (M_1 + M_2))**2 * G_cgs * (M_1 + M_2) * (2./(rr) - 1./a) / c_cgs / c_cgs, 0.9999999999)
        gam1 = 1. / np.sqrt(1.- (vorb1_sqr))
        gam2 = 1. / np.sqrt(1.- (vorb2_sqr))
        
        ## FULL DOPPLER FACTOR
        Dop1 = 1. / (gam1 * (1.- vr1 / c_cgs))**(3. - self.alphanu)
        Dop2 = 1. / (gam2 * (1.- vr2 / c_cgs))**(3. - self.alphanu)

        for j in len(Z1):            
            if Z1[j] > 0: 
                ##DD lensing secondary disk by primary BH
                D_s[j] = -Z2[j] 
                D_l[j] = -Z1[j] 
                M_s[j] = M_2
                M_l[j] = M_1
                # Calculate angle between lens and source
                dy = np.abs(Y1 - Y2)
                dx = np.abs(X1 - X2)
                dydx = np.sqrt(dx**2 + dy**2)
                D_ls = D_s - D_l # Distance between lens and source
                # Calculate r_E, theta
                r_E = np.sqrt(4. * G_cgs * M_l * D_ls / c_cgs**2)
                u = dydx / r_E
                Mlens = np.nan_to_num( (u**2 + 2.) / (u * np.sqrt(u**2 + 4.)) )
                izero = np.where(Mlens==0.0)[0] ##nan_to_num turns nans and infs into 0.0, so make these 1.0
                Mlens[izero] = 1.0
                magnification[j] = (1. - fs) * Dop1[j] + fs * Dop2[j] * Mlens[j]
            elif Z2[j] > 0:
                ##DD lensing primary disk
                D_s[j] = -Z1[j]
                D_l[j] = -Z2[j]
                M_s[j] = M_1
                M_l[j] = M_2
                # Calculate angle between lens and source
                dy = np.abs(Y1 - Y2)
                dx = np.abs(X1 - X2)
                dydx = np.sqrt(dx**2 + dy**2)
                D_ls = D_s - D_l # Distance between lens and source
                # Calculate r_E, theta
                r_E = np.sqrt(4 * G_cgs * M_l * D_ls / c_cgs**2)
                u = dydx / r_E
                Mlens = np.nan_to_num( (u**2 + 2.) / (u * np.sqrt(u**2 + 4.)) )
                izero = np.where(Mlens==0.0)[0]
                Mlens[izero] = 1.0
                magnification[j] = (1.-fs)*Dop1[j]*Mlens[j] + fs*Dop2[j]
            else:
                print('\007') 
            # j = j + 1
        return magnification

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
                           inclination_deg:float=0.0,
                           barycenter_velocity_c:float=0.0, 
                           argument_of_pericenter_deg:float=0.0,
                           spectral_slope_lnln:float=-1.0,
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
    barycenter_velocity_c (optional, default=0.0):
        line-of-sight velocity of the system barycenter in units of c
    argumnet_of_pericenter_deg (optional, default=0.0):
        argument of pericenter for eccentric binary orbit (see D'Orazio, Duffell & Tiede 2024)
    spectral_slope_lnln (optional, default=-1.0):
        dlog(Fnu)/dlog(nu) of the emitting spectrum in the observing band for boosting

    Return
    ------
    (ndarry) normalized periodic flux timeseries with same shape as accretion_series.primary/secondary/total in cgs
    """
    acc = accretion_series
    bad = BinaryAlphaDisk(acc.ecc, 
                          period, 
                          total_mass_msun, 
                          luminosity_distance_pc, 
                          eddington_ratio, 
                          accretion_efficiency, 
                          md_inner_edge_risco, 
                          cbd_inner_edge_a, 
                          cbd_outer_edge_a, 
                          inclination_deg,
                          barycenter_velocity_c, 
                          argument_of_pericenter_deg,
                          spectral_slope_lnln,
                        )
    return normazlied_flux_series_from_bad(frequency, acc, bad)

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
                         inclination_deg:float=0.0,
                         barycenter_velocity_c:float=0.0, 
                         argument_of_pericenter_deg:float=0.0,
                         spectral_slope_lnln:float=-1.0,
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
    inclination_deg_deg (optional, default=0.0):
        viewing inclination for the coplanar binary-disk system in degrees
    barycenter_velocity_c (optional, default=0.0):
        line-of-sight velocity of the system barycenter in units of c
    argumnet_of_pericenter_deg (optional, default=0.0):
        argument of pericenter for eccentric binary orbit (see D'Orazio, Duffell & Tiede 2024)
    spectral_slope_lnln (optional, default=-1.0):
        dlog(Fnu)/dlog(nu) of the emitting spectrum in the observing band for boosting

    Return
    ------
    (ndarry) periodic flux timeseries with same shape as accretion_series.primary/secondary/total in cgs
    """
    acc = accretion_series
    bad = BinaryAlphaDisk(acc.ecc, 
                          period, 
                          total_mass_msun, 
                          luminosity_distance_pc, 
                          eddington_ratio, 
                          accretion_efficiency, 
                          md_inner_edge_risco, 
                          cbd_inner_edge_a, 
                          cbd_outer_edge_a, 
                          inclination_deg,
                          barycenter_velocity_c, 
                          argument_of_pericenter_deg,
                          spectral_slope_lnln,
                        )
    return bad.fnu_total(frequency) * normazlied_flux_series_from_bad(frequency, acc, bad)

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

def normazlied_flux_series_from_bad(frequency:float, accretion_series:AccretionSeries, bad:BinaryAlphaDisk, lense_boost=False):
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
    mag1 = 1.0
    mag2 = 1.0
    if lense_boost:
        mag1 = bad.lensing_boosting_magnification(fs=0)
        mag2 = bad.lensing_boosting_magnification(fs=1)
    disk_flux = 1.0 - chi1 - chi2
    mdot_mean = np.mean(acc.total)
    return chi1 * acc.primary / mdot_mean * mag1 + chi2 * acc.secondary / mdot_mean * mag2 + disk_flux

def periodic_flux_series_from_bad(frequency:float, accretion_series:AccretionSeries, bad:BinaryAlphaDisk, lense_boost=False):
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
    return bad.fnu_total(frequency) * normazlied_flux_series_from_bad(frequency, accretion_series, bad, lense_boost=lense_boost)

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

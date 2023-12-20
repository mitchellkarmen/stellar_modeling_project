import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from matplotlib import pyplot as plt
import pandas as pd
from constants import *
import yaml


### GLOBAL VARS
with open('config.yaml') as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

M = config['stellarMass'] * Ms
composition = config['composition']
X = composition['X']
Y = composition['Y']
Z = composition['Z']
X_CNO = (composition['X_C'] + composition['X_N'] + composition['X_O'])*Z

def calc_mu(X):
    """
    Hansen. equation 1.55, approximation
    """
    return 4 / (3 + 5*X)

mu = calc_mu(X)

#### General utils
def get_density(T, P):
    """
    Calculate density given temperature and pressure.

    Parameters:
    - T (float): Temperature (K).
    - P (float): Pressure (erg/cm^3).

    Returns:
    - density (float): Density (g/cm^3).
    """
    P_rad = (1/3)*a*(T**4)
    density = (P - P_rad)*mu/(Na * k*T)
    return density



#### UTILS TO WORK WITH THE OPACITY TABLE

def load_opacity_table(filepath='GN93hz.txt'):
    """
    Load opacity table from file.

    Parameters:
    - filepath (str, optional): Path to the opacity table file (default is 'GN93hz.txt').

    Returns:
    - tab (pd.DataFrame): Opacity table.
    """

    n_lines = 9943
    lines_list = np.arange(n_lines)
    table_lines = np.arange(5789, 5861) # table 73
    lines_to_skip = [l for l in lines_list if l not in table_lines]
    tab = pd.read_csv(filepath, skiprows = lines_to_skip, delim_whitespace=True)

    return tab

opacity_table = load_opacity_table()

def make_interpolater(opacity_table):
    """
    Create an interpolator for the opacity table.

    Parameters:
    - opacity_table (pd.DataFrame): Opacity table.

    Returns:
    - r (RGI): Interpolator.
    """
    x = opacity_table.columns[1:].values.astype(float)
    y = opacity_table['logT'].values

    z = np.array([opacity_table[i].values for i in opacity_table.columns[1:]])
    r = RGI(points=(x, y), values=z, method='linear', bounds_error=False)

    return r

interpolater = make_interpolater(opacity_table)

def interp_opacity_table(rho, T):
    """
    Interpolate opacity from the opacity table.

    Parameters:
    - rho (float): Density (g/cm^3).
    - T (float): Temperature (K).

    Returns:
    - kappa (float): Opacity.
    """
    T_6 = T / 1e6
    R = rho / (T_6**3)
    log_R = np.log10(R)
    #print('logR = ', log_R)
    # if log_R < -8:
    #     log_R = -8
    # if log_R > 1:
    #     log_R = 1

    kappa = 10**interpolater((log_R, np.log10(T)))
    return kappa

### UTILS TO CALCULATE ENERGY GENERATION

def cno_energy(T, rho):
    """
    Calculate the energy generation from the CNO cycle.

    Parameters:
    - T (float): Temperature in Kelvin.
    - rho (float): Density in g/cm^3.

    Returns:
    - e_CNO (float): Energy generation from the CNO cycle.
    """
    T9 = T/1e9
    coeff = 8.24e25
    exp_coeff = -15.231
    g14 = 1 - (2.00*T9) + (3.41 * T9**2) - (2.43 * T9**3)

    e_CNO = coeff * g14 * X_CNO * X * rho * (T9**(-2/3)) * np.exp(exp_coeff * (T9**(-1/3)) - (T9/0.8)**2)
    return e_CNO

def get_phi(T7):
    """
    Get the phi factor for the PP energy generation.

    Parameters:
    - T7 (float): Temperature factor in units of 1e7 K.

    Returns:
    - phi (float): Phi factor.
    """
    # by eye...
    if T7 <= 1:
        return 1
    elif 1 < T7 < 2:
        phi = (0.4*T7) + 0.6
        return phi
    elif 2 <= T7< 3:
        return 1.4
    elif 3 <= T7:
        return 1.5
    else:
        print('uh oh!', T7)
        #raise ValueError('T7 is {}'.format(T7))
        return np.nan

def pp_energy(T, rho):
    """
    Calculate the energy generation from the PP chain.

    Parameters:
    - T (float): Temperature in Kelvin.
    - rho (float): Density in g/cm^3.

    Returns:
    - e_PP (float): Energy generation from the PP chain.
    """
    T9 = T/1e9
    T7 = T/1e7
    #print('PP: t9 and t7 are', T9, T7)
    g11 = 1 + (3.82 * T9) +(1.51 * T9**2) + (0.144 * T9**3) - (0.0114 * T9**4)
    f11 = np.exp(5.92e-3 * (rho / (T7**3))**(1/2))
    phi = get_phi(T7)
    if np.isnan(phi):
        return np.nan
    coeff = 2.57e4
    exp_coeff = -3.381

    e_PP = coeff * phi * f11 * g11 * rho * (X**2) * (T9**(-2/3)) * np.exp(exp_coeff / (T9**(1/3)))
    return e_PP


def energy_generation(T, rho):
    """
    Calculate the total energy generation.

    Parameters:
    - T (float): Temperature in Kelvin.
    - rho (float): Density in g/cm^3.

    Returns:
    - total_energy (float): Total energy generation.
    """
    e_PP = pp_energy(T, rho)
    e_CNO = cno_energy(T, rho)
    return e_PP + e_CNO

#### BOUNDARY CONDITIONS

def load1(P_c, T_c, M_r):
    """
    Calculate boundary conditions for the stellar core.

    Parameters:
    - P_c (float): Core pressure in erg/cm^3.
    - T_c (float): Core temperature in Kelvin.
    - M_r (float): Fractional mass.

    Returns:
    - boundaries (list): List containing [l, P, r, T].
    """
    rho_c = get_density(T_c, P_c)
    P = P_c - (3*G / (8*np.pi)* ((4*np.pi * rho_c/3) **(4/3)) * (M_r**(2/3)))
    e_c = energy_generation(T_c, rho_c)
    if np.isnan(e_c):
        return np.nan
    l = e_c * M_r
    r = (3*M_r/(4*np.pi*rho_c))**(1/3)
    # temperature depends on energy transport
    #print('rho, T', rho_c, T_c)
    kappa_c = interp_opacity_table(rho_c, T_c)
    # print('kappa_c is', kappa_c)
    del_rad = (3 / (16*np.pi*a*c)) * (P_c*kappa_c / (T_c**4)) * (l / (G * M_r))
    del_ad = 0.4
    if del_rad <= del_ad:
        # radiative transport
        print('core guess is radiative')
        del_actual = del_rad
        numerator = (3**(2/3)) * kappa_c * e_c * (rho_c**(4/3)) * (M_r**(2/3))
        denominator = 2 * a * c * ((4 * np.pi)**(2/3))
        T_quad = (T_c**4) - (numerator/denominator)
        T = T_quad**(1/4)
    elif del_rad > del_ad:
        # convective transport
        print('core guess is convective')
        del_actual = del_ad
        numerator = (np.pi**(1/3)) * G * del_actual * (rho_c**(4/3)) * (M_r**(2/3))
        denominator = (6**(1/3)) * P_c
        lnT = np.log(T_c) - (numerator/denominator)
        T = np.exp(lnT)
    else:
        print('something went wrong, interpolated outside possible T')
        print('del rad', del_rad)
        T = np.nan
        # raise ValueError('T is nan')
    return [l, P, r, T]


def load2(L_star, R_star):
    """
    Calculate boundary conditions for the stellar surface.

    Parameters:
    - L_star (float): Stellar luminosity in erg/s.
    - R_star (float): Stellar radius in cm.

    Returns:
    - boundaries (list): List containing [l, P, r, T] for the stellar surface.
    """
    l = L_star
    r = R_star
    sigma = a*c/4
    T = ( l / (4*np.pi*sigma*(r**2)) )**(1/4)

    # need better model of rho/kappa at surface
    rho_surf = 5e-7
    kappa_surf = interp_opacity_table(rho_surf, T)
    P = (2*G*M) / (3*kappa_surf*(r**2))
    return [l, P, r, T]


#### ODEs

def derivs(mass, params):
    """
    Calculate derivatives for the ODEs governing stellar structure.

    Parameters:
    - mass (float): Mass coordinate.
    - params (list): List containing [l, P, r, T].

    Returns:
    - derivatives (list): List containing [dldm, dPdm, drdm, dTdm].
    """
    l, P, r, T = params
    # no negative temperatures:
    # if T <= 0:
    #     T = 1
    rho = get_density(T, P)
    # if rho <= 0:
    #     rho = 0.01
    dldm = energy_generation(T, rho)
    dPdm = -G*mass / (4*np.pi*(r**4))
    drdm = (4 * np.pi * (r**2) * rho)**(-1)
    kappa = interp_opacity_table(rho, T)
    if np.isnan(kappa):
        return [np.nan, np.nan, np.nan, np.nan]
    del_rad = (3 / (16*np.pi*a*c)) * (P*kappa / (T**4)) * (l / (G * mass))
    del_ad = 0.4
    del_actual = np.minimum(del_rad, del_ad)
    dTdm = -1 * G * mass * T * del_actual / (4*np.pi * (r**4) * P)
    return [dldm, dPdm, drdm, dTdm]

def get_dels(T, P, kappa, l, mass):
    """
    gets the gradient given
    - T
    - P
    - kappa
    - mass

    returns [del_rad, del_ad, del_actual]
    """
    del_rad = (3 / (16*np.pi*a*c)) * (P*kappa / (T**4)) * (l / (G * mass))
    del_ad = 0.4
    del_actual = np.minimum(del_rad, del_ad)
    return [del_rad, del_ad, del_actual]


def load_guesses():
    """
    Generate initial guesses for the stellar core and surface conditions.

    Returns:
    - x0 (list): Initial guesses for [Pc, Tc, Lstar, Rstar].
    """
    Lstar_guess = Ls * ((M/Ms)**3.2)
    Rstar_guess = Rs * ((M/Ms)**0.75)

    Pc_guess = 80*(3*G*(M**2)) / (8*np.pi*(Rstar_guess**4)) # 50 works best so far
    Tc_guess = 1.5*(G*M*mu) / (2*Rstar_guess*Na*k)

    x0 = [Pc_guess, Tc_guess, Lstar_guess, Rstar_guess]
    return x0
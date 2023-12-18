import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import pandas as pd
import yaml


### GLOBAL VARS
with open('config.yaml') as f:
		config = yaml.load(f, Loader=yaml.FullLoader)

M = config['stellarMass']
XYZ = config['composition']
X = composition['X']
Y = composition['Y']
Z = composition['Z']
X_CNO = (composition['X_C'] + composition['X_N'] + composition['X_O'])*Z

#### UTILS TO WORK WITH THE OPACITY TABLE

def load_opacity_table(filepath='GN93hz.txt'):
    """
	need to update to work for arbitrary X, Y, Z. at the moment only load table 73
    """

    n_lines = 9943
    lines_list = np.arange(n_lines)
    table_lines = np.arange(5789, 5861) # table 73
    lines_to_skip = [l for l in lines_list if l not in table_lines]
    tab = pd.read_csv(filepath, skiprows = lines_to_skip, delim_whitespace=True)

    return tab

def interp_opacity_table(rho, T):
	"""
	would be faster if i removed the interpolator so I didn't have to re-fit it every time
	"""
	from scipy.interpolate import RegularGridInterpolator as RGI

    T_6 = T / 1e6
    R = rho / (T_6**3)
    log_R = np.log10(R)
    x = opacity_table.columns[1:].values.astype(float)
    y = opacity_table['logT'].values

    z = np.array([opacity_table[i].values for i in opacity_table.columns[1:]])
    r = RGI(points=(x, y), values=z, method='linear', bounds_error=False)

    return r((log_R, np.log10(T)))

### UTILS TO CALCULATE ENERGY GENERATION

def cno_energy(T, rho):
    T9 = T/1e9
    coeff = 8.24e25
    exp_coeff = -15.231
    g14 = 1 - (2.00*T9) + (3.41 * T9**2) - (2.43 * T9**3)

    e_CNO = coeff * g14 * X_CNO * X * rho * (T9**(-2/3)) * np.exp(exp_coeff * (T9**(-1/3)) - (T9/0.8)**2)
    return e_CNO

def get_phi(T7):
    # by eye...
    if T7 <= 1:
        return 1
    elif 1 < T7 < 2:
        T7 = (0.4*T7) + 0.6
    elif 2 <= T7< 3:
        return 1.4
    elif 3 <= T7:
        return 1.5

def pp_energy(T, rho):
    T9 = T/1e9
    T7 = T/1e7
    g11 = 1 + (3.82 * T9) +(1.51 * T9**2) + (0.144 * T9**3) - (0.0114 * T9**4)
    f11 = np.exp(5.92e-3 * (rho / (T7**3)))
    phi = get_phi(T7)
    coeff = 2.57e4
    exp_coeff = -3.381

    e_PP = coeff * phi * f11 * g11 * rho * (X**2) * (T9**(-2/3)) * np.exp(exp_coeff / (T9**(1/3)))
    return e_PP

def energy_generation(T, rho):

    e_PP = pp_energy(T, rho)
    e_CNO = cno_energy(T, rho)
    return e_PP + e_CNO


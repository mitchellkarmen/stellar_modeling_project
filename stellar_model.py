from shooting_fn import *
from stellar_utils import *
from constants import Ms
import scipy
from matplotlib import pyplot as plt

def run_model(bounds, M_fit=0.5):
    """
    Run a stellar model simulation.

    Parameters:
    - bounds (tuple): A tuple containing the following parameters in order: Pc (central pressure),
                      Tc (central temperature), L_star (stellar luminosity), R_star (stellar radius).
    - M_fit (float, optional): Fraction of the total mass (default is 0.5) at which the inner and outer integrations meet.

    Returns:
    - result (dict): A dictionary containing the simulation results.
        - 'mass' (ndarray): Array of mass values.
        - 'luminosity' (ndarray): Array of luminosity values.
        - 'pressure' (ndarray): Array of pressure values.
        - 'radius' (ndarray): Array of radius values.
        - 'temperature' (ndarray): Array of temperature values.

    Note:
    - The simulation is performed by integrating the stellar structure equations from the inner boundary
      (defined by Pc, Tc, and a fraction of the total mass) to the outer boundary (defined by L_star and R_star).
    - The total mass is divided into two regions: inner (from M_r to M_fit) and outer (from M to M_fit).
    """
    Pc, Tc, L_star, R_star = bounds
    M_r = 1e-8*M
    M_fit = M_fit*M
    inner_boundary = load1(Pc, Tc, M_r)
    outer_boundary = load2(L_star, R_star)
    n_steps = 1e4
    M_inner = np.linspace(M_r, M_fit, int(n_steps/2))
    M_outer = np.linspace(M, M_fit, int(n_steps/2))
    inner_result = scipy.integrate.solve_ivp(derivs, [M_r, M_fit], inner_boundary, t_eval=M_inner)
    outer_result = scipy.integrate.solve_ivp(derivs, [M, M_fit], outer_boundary, t_eval=M_outer)

    result = {}
    M_tot = np.concatenate([M_inner, np.flip(M_outer)])
    result['mass'] = M_tot
    l_inner, P_inner, r_inner, T_inner = inner_result.y
    l_outer, P_outer, r_outer, T_outer = outer_result.y
    result['luminosity'] = np.concatenate([l_inner, np.flip(l_outer)])
    result['pressure'] = np.concatenate([P_inner, np.flip(P_outer)])
    result['radius'] = np.concatenate([r_inner, np.flip(r_outer)])
    result['temperature'] = np.concatenate([T_inner, np.flip(T_outer)])
    return result

def make_plot(result, save=True):
    """
    Create and display a plot of normalized stellar model parameters.

    Parameters:
    - result (dict): A dictionary containing the simulation results.
        - 'mass' (ndarray): Array of mass values.
        - 'temperature' (ndarray): Array of temperature values.
        - 'pressure' (ndarray): Array of pressure values.
        - 'radius' (ndarray): Array of radius values.
        - 'luminosity' (ndarray): Array of luminosity values.
    - save (bool, optional): If True, save the plot as 'stellar_model_result.pdf' (default is True).
    """
    norm_mass = result['mass'] / Ms
    norm_temp = result['temperature'] / np.max(result['temperature'])
    norm_pressure = result['pressure'] / np.max(result['pressure'])
    norm_radius = result['radius'] / np.max(result['radius'])
    norm_luminosity = result['luminosity'] / np.max(result['luminosity'])
    plt.plot(norm_mass, norm_temp, label = 'Temperature')
    plt.plot(norm_mass, norm_pressure, label = 'Pressure')
    plt.plot(norm_mass, norm_radius, label = 'Radius')
    plt.plot(norm_mass, norm_luminosity, label = 'Luminosity')
    plt.yscale('log')
    plt.ylim(1e-5, 1.2)
    # plt.xscale('log')
    plt.legend()
    plt.xlabel('Mass ($M_{\odot}$)')
    plt.ylabel('Normalized Parameter')
    if save:
        plt.savefig('stellar_model_result.pdf')
    plt.show()

def make_file(result):
    """
    Create a CSV file containing additional calculated parameters based on the stellar model results.

    Parameters:
    - result (dict): A dictionary containing the simulation results.
        - 'mass' (ndarray): Array of mass values.
        - 'temperature' (ndarray): Array of temperature values.
        - 'pressure' (ndarray): Array of pressure values.
        - 'radius' (ndarray): Array of radius values.
        - 'luminosity' (ndarray): Array of luminosity values.

    Returns:
    - result_df (DataFrame): Pandas DataFrame containing the original simulation results and additional parameters.

    Note:
    - The function calculates density, opacity, energy generation rate, radiative gradient, adiabatic gradient,
      actual gradient, and energy transport type for each data point in the simulation.
    - The results are added as additional columns to the DataFrame.
    - The DataFrame is saved as a CSV file named 'star_result.csv'.
    """
    result_df = pd.DataFrame(result)
    density_list = np.zeros(len(result_df))
    opacity_table = load_opacity_table()
    interpolater = make_interpolater(opacity_table)
    kappa_list = np.zeros(len(result_df))
    energy_list = np.zeros(len(result_df))
    del_ad_list = np.zeros(len(result_df))
    del_rad_list = np.zeros(len(result_df))
    del_actual_list = np.zeros(len(result_df))
    energy_transport_list = np.full(len(result_df), 'radiative')
    for i, row in result_df.iterrows():
        density = get_density(row['temperature'], row['pressure'])
        energy = energy_generation(row['temperature'], density)
        kappa = interp_opacity_table(density, row['temperature'])
        del_rad, del_ad, del_actual = get_dels(row['temperature'], row['pressure'], kappa, row['luminosity'], row['mass'])
        if del_ad == del_actual:
            energy_transport_list[i] = 'convective'
        density_list[i] = density
        kappa_list[i] = kappa
        energy_list[i] = energy
        del_rad_list[i] = del_rad
        del_ad_list[i] = del_ad
        del_actual_list[i] = del_actual

    result_df['opacity'] = kappa_list
    result_df['density'] = density_list
    result_df['energy_generation_rate'] = energy_list
    result_df['radiative_grad'] = del_rad_list
    result_df['adiabatic_grad'] = del_ad_list
    result_df['actual_grad'] = del_actual_list
    result_df['transport'] = energy_transport_list

    result_df.to_csv('star_result.csv')
    return result_df



if __name__ == '__main__':
    initial_guess = load_guesses()
    print(initial_guess)

    converged_boundary = newton_root(shootf, initial_guess, max_iter=500)
    print(converged_boundary)

    result = run_model(converged_boundary)
    make_plot(result, save=False)
    make_file(result)
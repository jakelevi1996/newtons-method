"""
Main script for the repo, which runs experiments for different optimisers and
objective functions (EG comparing convergence time vs dimension for different
optimisers), and outputs the results as image files and/or to stdout.

TODO: include function for comparing the results of different optimisers
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import optimisers, objectives

def mesh(nx0=100, x0lim=[-3, 3], nx1=100, x1lim=[-2, 2]):
    """
    Construct a 2D mesh in the form of an array of 2D coordinates that can be
    iterated through and operated on, given upper and lower limits and the
    number of points in each dimension, and return also 2 1D arrays that can be
    used for the x and y arguments of a plotting function such as plt.pcolormesh
    """
    x0, x1 = np.linspace(*x0lim, nx0), np.linspace(*x1lim, nx1)
    x0_mesh, x1_mesh = np.meshgrid(x0, x1)
    x_list = np.stack([x0_mesh.ravel(), x1_mesh.ravel()], axis=1)
    return x_list, x0, x1

def plot_func_grad_curvature(
    objective, nx0=100, x0lim=[-3, 3], nx1=100, x1lim=[-2, 2],
    name="Objective function gradients", dir="Temp", file_ext="png"
):
    """
    Plot a 2D objective function, as well as its gradient norm, and the largest
    and smallest eignevalues of its Hessian
    """
    x_list, x0, x1 = mesh(nx0, x0lim, nx1, x1lim)
    f_list, g_list = np.empty(x_list.shape[0]), np.empty(x_list.shape[0])
    e1_list, e2_list = np.empty(x_list.shape[0]), np.empty(x_list.shape[0])
    # Calcualte objective function and its gradients and eigenvalues
    for i, x in enumerate(x_list):
        f_list[i] = objective(x)
        grad = objective.dfdx(x)
        g_list[i] = np.linalg.norm(grad)
        hess = objective.d2fdx2(x)
        evals = np.linalg.eigvalsh(hess)
        e1_list[i], e2_list[i] = max(evals), min(evals)
    
    # Initialise subplots
    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(16, 9)

    p_list = [f_list, g_list, e1_list, e2_list]
    name_list = ["Objective function", "Gradient norm",
        "Largest Hessian eigenvalue", "Smallest Hessian eigenvalue"]
    
    # Plot each subplot
    for z, fig_name, ax in zip(p_list, name_list, axes.ravel()):
        p = ax.pcolormesh(x0, x1, z.reshape(nx1, nx0))
        ax.set_title(fig_name)
        fig.colorbar(p, ax=ax)

    # Format, save and close
    fig.suptitle(name, fontsize=18)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("{}/{}.{}".format(dir, name, file_ext))
    plt.close(fig)

def smudge_plot(
    objective, optimiser, name="Smudge plot", n_lines=5,
    nx0_bg=200, nx1_bg=200, nx0_sm=10, nx1_sm=10,
    x0lims=[-3, 3], x1lims=[-2, 2], verbose=True, dir="Images/Smudge plots",
    file_ext="png"
):
    """
    Given a 2D objective function and an optimiser, and the details for a
    background grid and a smudge grid, plot the objective function on the
    background grid, and for each point point on the smudge grid, plot the path
    taken by the optimiser, with increasing transparency during each step (to
    simulate ink which is being smudged)

    TODO: use optional linesearch (maybe requires line-search being in its own
    method, or just calling minimise, but need to suppress output)

    ... Or better yet, accept kwargs for the optimiser

    TODO: augment name with names of optimiser and objective function
    """
    # Calculate background values
    x_bg_list, x0_bg, x1_bg = mesh(nx0_bg, x0lims, nx1_bg, x1lims)
    f_list = np.empty(x_bg_list.shape[0])
    for i, x in enumerate(x_bg_list): f_list[i] = objective(x)

    # Create figure and plot background vlaues
    plt.figure(figsize=[8, 6])
    plt.pcolormesh(x0_bg, x1_bg, f_list.reshape(nx1_bg, nx0_bg))

    # Initialise transparency values, starting values and smudge points
    alpha_list = np.linspace(1, 0, n_lines, endpoint=False)
    x_sm_list, x0_sm, x1_sm = mesh(nx0_sm, x0lims, nx1_sm, x1lims)
    smudge_points = np.empty([n_lines + 1, 2])
    # Iterate through each starting point for the smudge plot
    for i, x in enumerate(x_sm_list):
        if verbose:
            print("Smudging point {} of {}".format(i+1, x_sm_list.shape[0]))
        smudge_points[0] = x
        # Given the starting point x, iterate through each of the line-segments
        for i in range(n_lines):
            # Calculate smudge segment
            smudge_points[i+1], _ = optimiser(objective, smudge_points[i])
            # Plot smudge segment
            plt.plot(smudge_points[i:i+2, 0], smudge_points[i:i+2, 1], "k",
                alpha=alpha_list[i])
    # Format, save and close
    plt.xlim(x0lims)
    plt.ylim(x1lims)
    plt.title(name)
    plt.savefig("{}/{}.{}".format(dir, name, file_ext))
    plt.close()

def make_smudge_plots(objective):
    smudge_plot(objective, lambda f, x: optimisers.gradient_descent(
        f, x, n_iters=1, line_search_flag=False, learning_rate=2e-1
    ), name="SGD smudge plot")
    smudge_plot(objective, lambda f, x: optimisers.gradient_descent(
        f, x, n_iters=1, line_search_flag=True, beta=0.99, alpha=0.2
    # ), name="SGD+LS smudge plot", n_lines=3)
    ), name="SGD+LS smudge plot", nx0_sm=6, nx1_sm=6, n_lines=2)
    smudge_plot(objective, lambda f, x: optimisers.generalised_newton(
        f, x, n_iters=1, line_search_flag=False, learning_rate=0
    ), name="GN smudge plot")
    smudge_plot(objective, lambda f, x: optimisers.rectified_newton(
        f, x, n_iters=1, line_search_flag=False, learning_rate=0
    ), name="RN smudge plot")

def plot_dimensional_efficiency(
    optimiser_list, ObjectiveClass, ndims_list, n_repeats=5, distance_ratio=1,
    convergence_value=1e-10, seed=0, name="Convergence time vs dimension",
    dir="Images/Dimensional efficiency", file_ext="png", alpha=0.5
):
    # TODO: make list of final times for each dimension, repeat and optimiser.
    # If optimisation fails, record nan. Find mean across repeats of non-nan
    # results, and plot non-nan mean curves on the scatter plot

    # Set random seed
    np.random.seed(seed)
    # Initialise figure, and lists for colours and legend handles
    plt.figure(figsize=[8, 6])
    colours = plt.get_cmap("hsv")(
        np.linspace(0, 1, len(optimiser_list), endpoint=False))
    handles = []
    # Iterate through dimensions and repeats
    for i_d, ndims in enumerate(ndims_list):
        for i_r in range(n_repeats):
            # Initialise random scale and starting point
            scale = np.abs(np.random.normal(size=ndims))
            scale /= np.linalg.norm(scale)
            x0 = np.random.normal(size=ndims)
            x0 *= distance_ratio / np.linalg.norm(x0)
            # Iterate through optimisers
            for i_o, optimiser in enumerate(optimiser_list):
                print("Testing dimension {}/{}, repeat {}/{}, "
                    "optimiser {}/{}...".format(i_d + 1, len(ndims_list),
                        i_r + 1, n_repeats, i_o + 1, len(optimiser_list)))

                # Perform minimisation
                _, result = optimiser(ObjectiveClass(scale), x0,
                    convergence_value)

                # If optimisation succeeded, then add result to plot
                if result.objective[-1] <= convergence_value:
                    lines = plt.loglog(ndims, result.times[-1],
                        color=colours[i_o], marker="o", alpha=alpha)
                else: print("Optimisation failed for {}".format(result.name))

                # Check if the first dimension and repeat of each optimiser
                if i_d == 0 and i_r == 0:
                    # Make a record of the legend details
                    handles.append(Line2D([], [], color=colours[i_o],
                        marker="o", alpha=alpha, label=result.name))

    # Format, save and close
    plt.grid(True)
    plt.title(name)
    plt.legend(handles=handles)
    plt.xlabel("Dimension")
    plt.ylabel("Time until convergence (s)")
    plt.savefig("{}/{}.{}".format(dir, name, file_ext))
    plt.close()

def compare_optimisers_dimensional_efficiency():
    n_dims_list = np.unique(np.logspace(0, 3, 10, dtype=np.int))
    optimiser_list = [
        lambda f, x, f_lim: optimisers.gradient_descent(f, x, f_lim=f_lim,
            line_search_flag=True, n_iters=np.inf, t_lim=5),
        lambda f, x, f_lim: optimisers.generalised_newton(f, x, f_lim=f_lim,
            line_search_flag=True, n_iters=np.inf, t_lim=5),
        lambda f, x, f_lim: optimisers.block_generalised_newton(f, x,
            f_lim=f_lim, line_search_flag=True, n_iters=np.inf, t_lim=5),
        lambda f, x, f_lim: optimisers.parallel_block_generalised_newton(f, x,
            f_lim=f_lim, line_search_flag=True, n_iters=np.inf, t_lim=5,
            block_size=3),
        lambda f, x, f_lim: optimisers.rectified_newton(f, x, f_lim=f_lim,
            line_search_flag=True, n_iters=np.inf, t_lim=5),
    ]
    plot_dimensional_efficiency(optimiser_list, objectives.Gaussian,
        n_dims_list, distance_ratio=3,
        name="Dimensional efficiency of different optimisers")

def compare_block_sizes_dimensional_efficiency_sequential():
    n_dims_list = np.unique(np.logspace(0, 3, 10, dtype=np.int))
    block_sizes = np.unique(np.logspace(0, 2, 5, dtype=np.int))
    optimiser_list = [(
        lambda f, x, f_lim, b=b: optimisers.block_generalised_newton(
            f, x, f_lim=f_lim, line_search_flag=True, n_iters=np.inf, t_lim=5,
            block_size=b, name="Block size = {}".format(b))
    ) for b in block_sizes]
    plot_dimensional_efficiency(optimiser_list, objectives.Gaussian,
        n_dims_list, distance_ratio=3,
        name="Dimensional efficiency of sequential block sizes")

def compare_block_sizes_dimensional_efficiency_parallel():
    n_dims_list = np.unique(np.logspace(0, 3, 10, dtype=np.int))
    block_sizes = np.unique(np.logspace(0, 2, 5, dtype=np.int))
    optimiser_list = [(
        lambda f, x, f_lim, b=b: optimisers.parallel_block_generalised_newton(
            f, x, f_lim=f_lim, line_search_flag=True, n_iters=np.inf, t_lim=5,
            block_size=b, name="Block size = {}".format(b))
    ) for b in block_sizes]
    plot_dimensional_efficiency(optimiser_list, objectives.Gaussian,
        n_dims_list, distance_ratio=3,
        name="Dimensional efficiency of parallel block sizes")

def compare_results_curves(results_list, ): raise NotImplementedError

if __name__ == "__main__":
    # print(mesh(nx0=3, nx1=5))
    # objective = objectives.Gaussian(scale=[1, 5])
    # objective = objectives.Cauchy(scale=[1, 5])
    # objective = objectives.SumOfGaussians()
    # plot_func_grad_curvature(objective)
    # make_smudge_plots(objective)
    compare_optimisers_dimensional_efficiency()
    # compare_block_sizes_dimensional_efficiency_sequential()
    # compare_block_sizes_dimensional_efficiency_parallel()

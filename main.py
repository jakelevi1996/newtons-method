"""
Main script for the repo, which runs experiments for different optimisers and
objective functions (EG comparing convergence time vs dimension for different
optimisers), and outputs the results as image files and/or to stdout.

TODO: include function for comparing the results of different optimisers
"""
import numpy as np
import matplotlib.pyplot as plt
import optimisers, objectives, plotting

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
    f_list = np.empty(x_list.shape[0])
    g_list = np.empty(x_list.shape[0])
    e1_list = np.empty(x_list.shape[0])
    e2_list = np.empty(x_list.shape[0])
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
    plt.savefig("{}/{}".format(dir, name))
    plt.close(fig)

def smudge_plot(
    objective, optimiser, name="Smudge plot", n_lines=5,
    nx0_bg=200, nx1_bg=200, nx0_sm=9, nx1_sm=9,
    x0lims=[-3, 3], x1lims=[-2, 2], verbose=True, dir="Temp"
):
    """
    Given a 2D objective function and an optimiser, and the details for a
    background grid and a smudge grid, plot the objective function on the
    background grid, and for each point point on the smudge grid, plot the path
    taken by the optimiser, with increasing transparency during each step (to
    simulate ink which is being smudged)

    TODO: use optional linesearch (maybe requires line-search being in its own
    method, or just calling minimise, but need to suppress output)
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
            smudge_points[i+1] = smudge_points[i] + optimiser.get_step(
                objective, smudge_points[i])[0]
            # Plot smudge segment
            plt.plot(smudge_points[i:i+2, 0], smudge_points[i:i+2, 1], "k",
                alpha=alpha_list[i])
    # Format, save and close
    plt.xlim(x0lims)
    plt.ylim(x1lims)
    plt.savefig("{}/{}".format(dir, name))
    plt.close()



if __name__ == "__main__":
    # print(mesh(nx0=3, nx1=5))
    objective = objectives.Gaussian(scale=[1, 5])
    plot_func_grad_curvature(objective)
    smudge_plot(objective, optimisers.GradientDescent(), name="SGD smudge plot")
    smudge_plot(objective, optimisers.GeneralisedNewton(
        learning_rate=1e-1, max_step=0.5), name="GN smudge plot")

"""
Module which contains various different classes for optimisation of a given
objective function (EG gradient descent, generalised Newton's method), and an
astract parent class from which they inherit, which implements a 2-way
line-search. Also includes a Result class, for storing the results of each
optimisation routine, and displaying progress and summaries of each optimisation
routine

TODO:
-   *Line-search forward tracking: until objective function no longer increases
-   Timeout and evaluations based on elapsed time, not number of iterations
-   Objective function evaluations performed in a separate process? Does this
    make things quicker?
-   Optimisers to include:
    -   Adam
    -   PSO
    -   Examples from Antoniou/Nocedal, EG conjugate gradients, L?
"""

import numpy as np
from time import perf_counter
import objectives


class Result():
    """
    Class to store the results of optimisation in a single object which can be
    passed directly to plotting and/or analysis functions, as well as methods
    for updating and displaying results
    """
    def __init__(self, name):
        """
        Store the name of the experiment (which is useful later when displaying
        results), display table headers, initialise lists for objective function
        evaluations and the time and iteration number for each evaluation, and
        record the start time for the results list
        """
        self.name = name if (name is not None) else "Unnamed experiment"
        self.display_headers()

        self.objective = []
        self.times = []
        self.iters = []
        self.start_time = perf_counter()
    
    def update(self, f, i):
        t = perf_counter() - self.start_time
        self.objective.append(f)
        self.times.append(t)
        self.iters.append(i)
    
    def display_headers(self):
        # num_fields, field_width = 3, 10
        print("\nPerforming test \"{}\"...".format(self.name))
        print(" | ".join(["{:10}"] * 3).format(
            "Iteration", "Time (s)", "Objective function"))
        print(" | ".join(["-" * 10] * 3))

    def display_last(self):
        print("{:10d} | {:10.2f} | {:10.7e}".format(
            self.iters[-1], self.times[-1], self.objective[-1]))

    
    def display_summary(self, n_iters):
        t_total = perf_counter() - self.start_time
        t_mean = t_total / n_iters
        print("-" * 50,
            "{:30} = {}".format("Test name", self.name),
            "{:30} = {:,.4f} s".format("Total time", t_total),
            "{:30} = {:,}".format("Total iterations", n_iters),
            "{:30} = {:.4f} ms".format("Average time per iteration",
                1e3 * t_mean),
            "{:30} = {:,.1f}".format("Average iterations per second",
                1 / t_mean),
            sep="\n", end="\n\n")
    
    def save(self, filename): raise NotImplementedError
    def load(self, filename): raise NotImplementedError


def backtrack_condition(s, f, x, delta, alpha, dfdx, f0):
    """
    Compares the actual reduction in the objective function to that which is
    expected from a first-order Taylor expansion. Returns True if a reduction in
    the step size is needed according to this criteria, otherwise returns False
    """
    reduction = f0 - f(x + s * delta)
    expected_reduction = -s * np.dot(delta, dfdx)
    return reduction < (alpha * expected_reduction)

class Minimiser():
    """
    Minimiser: interface for optimisation routines, containing code which is
    common to all minimisation methods, to be inherited by each child class
    (which should be specific to a particular minimisation method). Child
    classes should override the __init__ and get_step methods, whereas the
    minimise method is designed to be common to all optimisation methods.
    """
    def __init__(self, name, *args):
        """
        Initialisation method for the Minimiser interface. This method should be
        overridden by the child class and accept arguments specific to the type
        of optimisation (EG learning rate for gradient descent, max-step and
        learning rate for generalised-Newton), as well as a default name string
        """
        self.name = name
        raise NotImplementedError

    def get_step(self, objective, x):
        """
        Method to get the descent step during each iteration of minimisation;
        this should be overridden by each child class which implements a
        specific minimisation method
        """
        raise NotImplementedError

    def minimise(
        self, f, x0, n_iters=10000, print_every=500, line_search=False, s0=1,
        alpha=0.8, beta=0.5, final_backstep=False, name=None
    ):
        """
        minimisation method (EG optional line-search, recording results,
        displaying a final summary), and calls a get_step method, which should
        be implemented by each specific child class.
        """
        # Set initial parameters and start time
        x, s = x0.copy(), s0
        # If no name is provided, use the default
        if name is None: name = self.name
        # Initialise result object, including start time of iteration
        result = Result(name)
        for i in range(n_iters):
            if i % print_every == 0: #TODO: make this condition time-based
                # Evaluate the model
                result.update(f(x), i)
                result.display_last()
            
            # Update parameters
            step, dfdx = self.get_step(f, x)
            if line_search:
                backtrack_params = (f, x, step, alpha, dfdx, f(x))
            
                if backtrack_condition(s, *backtrack_params):
                    # Reduce step size until error reduction is good enough
                    s *= beta
                    while backtrack_condition(s, *backtrack_params): s *= beta
                else:
                    # Increase step size until reduction is not good enough

                    # TODO: I think there needs to be an option for "increase
                    # until the objective function decreases"; otherwise the
                    # forward tracking can skip over the maximum
                    s /= beta
                    while not backtrack_condition(s, *backtrack_params):
                        s /= beta
                    if final_backstep: s *= beta
                x += s * step
            else:
                x += step

        # Evaluate final performance
        result.update(f(x), n_iters)
        result.display_last()
        result.display_summary(n_iters)

        return result

class GradientDescent(Minimiser):
    """ Class for minimisation using simple gradient-descent """
    def __init__(self, learning_rate=1e-1, name="Gradient descent"):
        self.learning_rate = learning_rate
        self.name = name
    
    def get_step(self, objective, x):
        dfdx = objective.dfdx(x)
        return -self.learning_rate * dfdx, dfdx


class GeneralisedNewton(Minimiser):
    """
    Class for minimisation using Newton's method, generalised to non-convex
    objective functions using an eigendecomposition of the Hessian matrix
    """
    def __init__(
        self, learning_rate=1e-1, max_step=1, name="Generalised Newton"
    ):
        self.learning_rate = learning_rate
        self.max_step = max_step
        self.name = name
    
    def get_step(self, objective, x):
        # Get gradients of objective function
        grad = objective.dfdx(x)
        hess = objective.d2fdx2(x)
        # Rotate gradient into eigenbasis of Hessian
        evals, evecs = np.linalg.eigh(hess)
        grad_rot = np.matmul(evecs.T, grad)
        # Take a Newton step in directions in which this step is not too big
        step_rot = np.where(
            np.abs(grad_rot) > (np.abs(evals) * self.max_step),
            -self.learning_rate * grad_rot, -grad_rot / np.abs(evals)
        )
        # Rotate gradient back into original coordinate system and return
        return np.matmul(evecs, step_rot), grad

def compare_function_times(input_dict_list, n_repeats=5, verbose=True):
    """
    Perform multiple repeats of an experiment, and print the mean and STD of the
    results. input_dict_list should be a list of dictionaries, one for each
    function to be tested. Each dictionary should have func, args and name keys,
    for the function which is to be called, a tuple of args with which it is to
    be called, and the name which is to be printed along with the results.
    """
    t_list = np.empty([len(input_dict_list), n_repeats])
    # Iterate through each input dictionary
    for i, input_dict in enumerate(input_dict_list):
        # Iterate through each repeat
        for j in range(n_repeats):
            func, args = input_dict["func"], input_dict["args"]
            t0 = perf_counter()
            func(*args)
            t1 = perf_counter()
            # Record time taken
            t_list[i, j] = t1 - t0
    if verbose:
        hline = "\n" + "-" * 50 + "\n"
        print("{0}Funcion time comparison results{0}".format(hline), end="")
        # Print results for each input dictionary
        for input_dict, t in zip(input_dict_list, t_list):
            print("\n".join(["Function name: {:15}", "Mean time (s): {:.4f}",
                "STD time (s): {:.4f}"]).format(input_dict["name"], t.mean(),
                t.std()), sep="\n", end=hline)
    return t_list
        


if __name__ == "__main__":
    nits = 10000
    print_every = nits // 10
    lr, max_step = 1e-1, 1
    x0 = 2*np.array([1.0, 1.0, 1.0])
    f = objectives.Gaussian(scale=[1, 2, 3])
    # Do warmup experiment
    GradientDescent(lr, name="Warmup").minimise(f, x0, 500, 10)
    # Compare times
    input_dict_list = [
        {"func": GradientDescent(lr).minimise,
            "args": (f, x0, nits, print_every, True),
            "name": "Gradient descent"},
        {"func": GeneralisedNewton(lr, max_step).minimise,
            "args": (f, x0, nits, print_every),
            "name": "Generalised Newton"}
    ]
    compare_function_times(input_dict_list, n_repeats=3)
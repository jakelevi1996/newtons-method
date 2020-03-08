"""
Module which contains various different classes for optimisation of a given
objective function (EG gradient descent, generalised Newton's method), and an
astract parent class from which they inherit, which implements a 2-way
line-search. Also includes a Result class, for storing the results of each
optimisation routine, and displaying progress and summaries of each optimisation
routine

What is happening to GN + optimal LS when x0 = [10]*3 ?? Debug required

Generalised Newton now converges to 0.000e+00 from x0 = 2*np.array([1.0, 1.0,
1.0]), which is better than before, however we still have s converging to 2
instead of 1 (need to debug this to find out why). For x0 = 10*np.array([1.0, 1.0, 1.0]), we have only GN without LS
succeeding; need to compare this with both types of LS in the debugger to find
out why (especially optimal-step) LS is not succeeding in this scenario.

The functions in this module should ideally fulfil the following properties:
-   When doing line-search for generalised Newton's method, when getting close
    to the optimum, the step-size should converge to 1 (this should hopefully
    allow similar performance to having no line-search near the optimum)
    -   Could this be achieved by replacing the backtrack condition with
        "iterate backwards until the backtrack condition is fulfilled, AND the
        objective function stops decreasing"? This would probably mean the new
        objective function value should be calculated outside of the backtrack
        condition function and passed in as an argument
-   When getting close to the optimum, line-search calls should be very cheap
    (maximum 1 iteration per call), and forward-optimal should be cheaper per
    iteration close to the optimum than forward-backtrack
-   When starting far away from the optimum, line-search for generalised
    Newton's method should not be performing worse than no line-search
-   Should gradient-descent with line-search be able to succeed without the
    step-size diverging, when starting far away from the optimum?


TODO: Question: for Generalised Newton, why is forward-optimal line-search
> 4x slower than forward-backtrack line-search? Is it due to behaviour, EG the
step-size fluctuating? Can the line-search function be altered to change this?
NB forward-optimal uses final-backstep, which is not true for
forward-backtrack...
-   NB both forward-optimal and forward-backtrack both reach f(x)=0e+0 in the
    same time, however forward-optimal does so in fewer iterations (12 vs 40).
    So it must be the fact that whatever happens in later iterations for
    forward-optimal is less efficient than forward-backtrack?
-   Why does no line-search make it to x = 0, step = 0, even when convergence
    during the first 25 iterations is much slower? Why do the other methods not
    have s = 1 as x -> 0?
-       NB Generalised Newton is the only method that succeeds with x0 =
        10*np.array([1.0, 1.0, 1.0]), and does so after 607 iterations. Why is
        line-search performing worse than no line-search (IE always keeping s =
        1) in this situation?

TODO:
-   Use functions which have callable arguments get_step and (shared) minimise,
    instead of classes which inherit from Minimiser
    -   get_step functions could come from custom classes which define
        parameters in an __init__ method
-   *Line-search forward tracking: until objective function no longer increases
-   Turn Minimiser class into a _minimise private function, which takes a
    get_step callable argument; for each minimiser, implement a get_step
    function, and a wrapper for the _minimise function
-   Timeout and evaluations based on elapsed time, not number of iterations
-   Objective function evaluations performed in a separate process? Does this
    make things quicker?
-   Optimisers to include:
    -   Adam
    -   PSO
    -   Examples from Antoniou/Nocedal, EG conjugate gradients, LBGFS?
"""

import numpy as np
from numpy.linalg import norm
from time import perf_counter
import objectives
import warnings


class Result():
    """
    Class to store the results of optimisation in a single object which can be
    passed directly to plotting and/or analysis functions, as well as methods
    for updating and displaying results

    TODO: Make this class configurable, so columns such as step-size and |x| are
    optional, and the column width and format spec for each column is
    configurable. Also implement saving and loading of results
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
        self.step_size = []
        self.x_norm = []
        self.start_time = perf_counter()
    
    def update(self, i, f, s, x, verbose=True):
        t = perf_counter() - self.start_time
        self.objective.append(f)
        self.times.append(t)
        self.iters.append(i)
        self.step_size.append(s)
        self.x_norm.append(np.linalg.norm(x))
        if verbose: self.display_last()
    
    def display_headers(self):
        # num_fields, field_width = 3, 10
        print("\nPerforming test \"{}\"...".format(self.name))
        print(" | ".join(["{:10}"] * 5).format("Iteration", "Time (s)",
            "Objective", "Step size", "|x|"))
        print(" | ".join(["-" * 10] * 5))

    def display_last(self):
        print("{:10d} | {:10.3f} | {:10.3e} | {:10.3e} | {:10.3e}".format(
            self.iters[-1], self.times[-1], self.objective[-1],
            self.step_size[-1], self.x_norm[-1]))

    
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


def backtrack_condition(s, f_new, f_0, delta_dot_dfdx, alpha):
    """
    Compares the actual reduction in the objective function to that which is
    expected from a first-order Taylor expansion. Returns True if a reduction in
    the step size is needed according to this criteria, otherwise returns False.

    Added tolerance because if delta is very small, x + s * delta == x (within
    floating point arithmetic), => f0 == f(x + s * delta), => reduction == 0, =>
    return value is always true => infinite loop
    """
    # If s -> 0, then x_new -> x, in which case don't continue back-tracking
    # TODO: compare np.all(x_new == x) vs np.array_equal(x, x_new) for
    # performance, and also compare to not using any checking 
    reduction = f_0 - f_new
    if reduction == 0:
        return False
        # if np.all(x_new == x): return False
    # Alternatively: just check if s == 0?? What to do if s == 0?
    expected_reduction = -s * delta_dot_dfdx
    # return reduction + tol < (alpha * expected_reduction)
    return reduction < (alpha * expected_reduction)

def line_search(
    x, s, delta, f, dfdx, alpha, beta, 
    optimal_step, final_backstep
):
    """
    do line_search...

    How to handle the case that step or s is so small that x + s * delta == x?
    Prevent from infinite back-tracking by checking for change in x, prevent
    from infinite forward tracking by checking that s is finite, and if s
    diverges, then return the input s / beta, so at least after many iterations,
    x + s * delta != x, and x can start to move towards a minimum.

    TODO: this seems better, except for the following experiment:
    GradientDescent(lr, name="SGD, optimal forward LS").minimise(f, x0, 10000,
    print_every, True, forward_backtrack_condition=False, final_backstep=True)
    ... after 1023 iterations, s -> inf, |x| ~~ 1e9, |step| = 0; the step must
    be being followed, but x has diverged away from the minimum

    TODO: update comments, and compare performance of different combinations of
    forward_backtrack_condition and final_backstep; if there is a clear best
    choice for a range of different starting conditions, then remove redundant
    choices

    What TODO if s == 0?

    Whether to increase s at the end of forward tracking should depend on if
    x_new == x_old (first check if f_new == f_old? Do all this in a
    sub-function?)
    """
    f_0, s_old = f(x), s
    delta_dot_dfdx = np.dot(delta, dfdx)
    bt_params = (f_0, delta_dot_dfdx, alpha)
    f_new, f_old = f(x + s * delta), f_0

    if backtrack_condition(s, f_new, *bt_params):
        # Reduce step size until error reduction is good enough
        s *= beta
        f_new, f_old = f(x + s * delta), f_new
        if optimal_step:
            # When close to the optimum with small s, it is possible that even
            # if the backtrack condition is not met, back-tracking will still
            # lead to an increase in the objective function, so we want to keep
            # back-tracking until we don't need to back-track AND the objective
            # function is increasing
            while backtrack_condition(s, f_new, *bt_params) or f_new < f_old:
                s *= beta
                if s == 0:
                    warnings.warn(
                        "s has converged to 0; resetting t0 s_old/beta ...")
                    return s_old * beta
                f_new, f_old = f(x + s * delta), f_new

            # if final_backstep: s /= beta
            # if final_backstep and f_new > f_old: s *= beta
            # TODO 2020-08-03: change and to or?
            # if final_backstep and f_new > f_old: s /= beta
            if final_backstep or f_new > f_old: s /= beta
        else:
            while backtrack_condition(s, f_new, *bt_params):
                # print("\tBacktracking")
                s *= beta
                f_new = f(x + s * delta)
    else:
        # Increase step size until reduction is not good enough
        if optimal_step:
            # Track forwards until objective function stops decreasing
            s /= beta
            # if not np.isfinite(s):
            #     print("Debug")
            f_new = f(x + s * delta)
            while f_new < f_old:
                # print("\t", s, f_old)
                s /= beta
                # if np.max(np.abs(s*step) > 1e-3):
                #     print("Big s")
                if not np.isfinite(s):
                    warnings.warn(
                        "s has diverged; resetting t0 s_old/beta ...")
                    return s_old / beta
                f_new, f_old = f(x + s * delta), f_new
            # print(s)
        else:
            # Use same backtrack condition to track forwards
            s /= beta
            while not backtrack_condition(s, f_new, *bt_params):
                s /= beta
                f_new = f(x + s * delta)
                if not np.isfinite(s):
                    warnings.warn("s has diverged; resetting t0 s_old/beta ...")
                    return s_old / beta

        # if final_backstep: s *= beta
        # if final_backstep and f_new > f_old:
        if final_backstep or f_new > f_old: s *= beta

    return s

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
        self, f, x0, n_iters=10000, print_every=500, line_search_flag=False,
        s0=1.0, alpha=0.5, beta=0.5, final_backstep=False, name=None,
        optimal_step=False, verbose=False
    ):
        """
        minimisation method (EG optional line-search, recording results,
        displaying a final summary), and calls a get_step method, which should
        be implemented by each specific child class.

        TODO: run tests comparing forward_backtrack_condition True vs False, and
        remove worse option. Same for final_backstep
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
                result.update(i, f(x), s, x, verbose=verbose)
            
            # Update parameters
            delta, dfdx = self.get_step(f, x)
            # Check if delta = 0; if so, minimisation can't continue
            if not np.any(delta):
                warnings.warn(
                    "|delta| = 0 during iteration {}; exiting...".format(i))
                result.update(i, f(x), s, x, verbose=verbose)
                return x, result
            if line_search_flag:
                s = line_search(x, s, delta, f, dfdx, alpha, beta,
                    optimal_step, final_backstep)
                x += s * delta
            else:
                x += delta

        # Evaluate final performance
        result.update(n_iters, f(x), s, x, verbose=verbose)
        result.display_summary(n_iters)

        return x, result

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
        step_rot = np.where((self.max_step * np.abs(evals)) > np.abs(grad_rot),
            -grad_rot / np.abs(evals), -self.learning_rate * grad_rot)
        # Rotate gradient back into original coordinate system and return
        return np.matmul(evecs, step_rot), grad

def compare_function_times(input_dict_list, n_repeats=5, verbose=True):
    """
    Perform multiple repeats of an experiment, and print the mean and STD of the
    results. input_dict_list should be a list of dictionaries, one for each
    function to be tested. Each dictionary should have func, args and name keys,
    for the function which is to be called, a tuple of args with which it is to
    be called, and the name which is to be printed along with the results.

    Example usage:
    input_dict_list = [
        {"func": GradientDescent(lr).minimise,
            "args": (f, x0, nits, print_every, True),
            "name": "Gradient descent"},
        {"func": GeneralisedNewton(lr, max_step).minimise,
            "args": (f, x0, nits, print_every),
            "name": "Generalised Newton"}
    ]
    compare_function_times(input_dict_list, n_repeats=3)
    """
    t_list = np.empty([len(input_dict_list), n_repeats])
    # Iterate through each input dictionary
    for i, input_dict in enumerate(input_dict_list):
        # Get function and input
        func, args = input_dict["func"], input_dict["args"]
        # Iterate through each repeat
        for j in range(n_repeats):
            # Perform experiment
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
    # nits = 1000
    # x0 = 10*np.array([1.0, 1.0, 1.0])
    nits = 40
    x0 = 2*np.array([1.0, 1.0, 1.0])

    print_every = nits // 10
    # print_every = 1
    lr, max_step = 1e-1, 1
    f = objectives.Gaussian(scale=[1, 2, 3])

    # Do warmup experiment
    GradientDescent(lr, name="Warmup").minimise(f, x0, 500, 100, verbose=True)

    GradientDescent(lr, name="GD, optimal-step LS").minimise(f, x0, nits,
        print_every, True, optimal_step=True, verbose=True,
        # final_backstep=True)
        final_backstep=False)
    # GeneralisedNewton(lr, max_step).minimise(f, x0, nits, print_every, True,
    #     name="Generalised Newton, forward-backtrack LS")
    GeneralisedNewton(lr, max_step).minimise(f, x0, nits, print_every, True,
        name="Generalised Newton, optimal-step LS",
        optimal_step=True, final_backstep=True, verbose=True)
    # GeneralisedNewton(lr, max_step).minimise(f, x0, nits, print_every, False,
    #     name="Generalised Newton, no LS")
    
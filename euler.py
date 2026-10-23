import numpy as np

def euler_solve(f, t_span, y0, h):
    """
    Solves an initial value problem dy/dt = f(t, y) using Euler's Method.
    Supports both scalar ODEs and systems of ODEs.
    """
    if h <= 0:
        raise ValueError("Step size 'h' must be strictly positive.")
    t_start, t_end = t_span
    if t_end <= t_start:
        raise ValueError("t_end must be strictly greater than t_start.")
        
    t_grid = np.arange(t_start, t_end + h, h)
    N = len(t_grid)
    
    y0_arr = np.atleast_1d(np.array(y0, dtype=float))
    y_grid = np.zeros((N, len(y0_arr)))
    y_grid[0] = y0_arr
    
    for n in range(N - 1):
        t_n = t_grid[n]
        y_n = y_grid[n]
        y_grid[n + 1] = y_n + h * np.array(f(t_n, y_n))
        
    if np.isscalar(y0):
        return t_grid, y_grid.flatten()
    return t_grid, y_grid


if __name__ == "__main__":
    f_test = lambda t, y: -2 * y
    t_grid, y_grid = euler_solve(f_test, (0, 2), 1.0, 0.1)
    print("Euler solver execution successful! Test output length:", len(t_grid))
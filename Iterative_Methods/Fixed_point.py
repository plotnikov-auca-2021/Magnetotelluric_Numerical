def g(x):
    return x * (x - 1)


def fixed_point_iteration(x0, tol=0.01, max_iters=10000):
    i = 0
    while i < max_iters:
        x1 = x0 - 0.000001 * g(x0)
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1
        i += 1
    return None


x_initial_first = 10
first_root = fixed_point_iteration(x_initial_first)

if first_root:
    print(f"Root: {first_root}")
else:
    print("Root cannot be found within the maximum number of iterations.")

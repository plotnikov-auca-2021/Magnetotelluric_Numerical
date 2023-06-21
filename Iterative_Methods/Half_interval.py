def f(x):
    return (x ** 2) - x


def u(x):
    return x ** 2


def v(x):
    return x - ((x ** 2) / (2 * ((1 - x) ** 2)))


def half_interval_method_recursive(x0, x1, tol=0.001, max_iters=1000):
    c = (x0 + x1) / 2
    if f(c) == 0 or (x1 - x0) / 2 < tol:
        return c
    if f(c) * f(x0) > 0:
        return half_interval_method_recursive(c, x1, tol, max_iters - 1)
    else:
        return half_interval_method_recursive(x0, c, tol, max_iters - 1)


fx0 = -0.5
fx1 = 2
f_root = half_interval_method_recursive(fx0, fx1)

ux0 = -1
ux1 = 2
u_root = half_interval_method_recursive(ux0, ux1)

vx0 = -8.1
vx1 = 9.6
v_root = half_interval_method_recursive(vx0, vx1)

if f_root:
    print(f"Root of f(x): {f_root}")
else:
    print("Root cannot be found on this interval")

if u_root:
    print(f"Root of u(x): {u_root}")
else:
    print("Root cannot be found on this interval")

if v_root:
    print(f"Root of v(x): {v_root}")
else:
    print("Root cannot be found on this interval")

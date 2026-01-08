import mpmath as mp
import numpy as np
from scipy.optimize import differential_evolution, minimize
from decimal import Decimal, getcontext, ROUND_HALF_UP

mp.mp.dps = 80  # internal precision; plenty for 20 decimals

# --------------------------------------------------------------------
# INPUT: put the three zeros of P here (complex numbers)
roots = [1+0j, 6+0j, 0+3j]
# roots = [0+0j, 9+0j, 8+2j]
# roots = [1+1j, -12+0j, 0+3j]
# roots = [3+0j, -3+0j, 0+5j]
# roots = [3+0j, 6+0j, 0+2j]
# --------------------------------------------------------------------

def coeffs_from_roots(rts):
    r1, r2, r3 = map(mp.mpc, rts)
    s1 = r1 + r2 + r3
    s2 = r1*r2 + r1*r3 + r2*r3
    s3 = r1*r2*r3
    # P(z)=z^3 + a z^2 + b z + c
    a = -s1
    b = s2
    c = -s3
    return a, b, c

def P(a, b, c, z):
    return z**3 + a*z**2 + b*z + c

def saddle_points(a, b, c, z):
    """Find saddle points using numpy for robustness, then refine with mpmath."""
    # (u-z)P'(u) - P(u) = 0  ==>  2 u^3 + (a-3z)u^2 - 2 a z u - (b z + c)=0
    poly = [2, a - 3*z, -2*a*z, -(b*z + c)]
    
    # Convert to complex128 for numpy
    poly_np = [complex(p) for p in poly]
    np_roots = np.roots(poly_np)
    
    # Refine each root with mpmath using Newton iteration
    def refine_root(u0, poly_coeffs):
        u = mp.mpc(u0)
        for _ in range(50):
            # Evaluate polynomial and derivative at u
            p_val = poly_coeffs[0]*u**3 + poly_coeffs[1]*u**2 + poly_coeffs[2]*u + poly_coeffs[3]
            dp_val = 3*poly_coeffs[0]*u**2 + 2*poly_coeffs[1]*u + poly_coeffs[2]
            if abs(dp_val) < 1e-100:
                break
            u_new = u - p_val / dp_val
            if abs(u_new - u) < mp.mpf(10)**(-70):
                break
            u = u_new
        return u
    
    us = [refine_root(r, poly) for r in np_roots]
    return sorted(us, key=lambda u: (mp.re(u), mp.im(u)))

def height(a, b, c, z, u):
    return mp.log(abs(P(a, b, c, u))) - mp.log(abs(u - z))

def vertex_degree3(rts):
    """Find the degree-3 vertex (triple saddle tie point) using global optimization."""
    a, b, c = coeffs_from_roots(rts)
    
    # Convert coefficients to complex for numpy operations
    a_np, b_np, c_np = complex(a), complex(b), complex(c)
    
    def saddle_points_np(z):
        poly = [2, a_np - 3*z, -2*a_np*z, -(b_np*z + c_np)]
        return np.roots(poly)
    
    def P_np(z):
        return z**3 + a_np*z**2 + b_np*z + c_np
    
    def height_np(z, u):
        return np.log(abs(P_np(u))) - np.log(abs(u - z))
    
    def spread_np(xy):
        x, y = xy
        z = complex(x, y)
        try:
            us = saddle_points_np(z)
            hs = [height_np(z, u) for u in us]
            m = sum(hs)/3
            return sum((h-m)**2 for h in hs)
        except:
            return 1e10  # Return large value on error
    
    # Get search bounds based on roots
    rs = [mp.re(mp.mpc(r)) for r in rts]
    is_ = [mp.im(mp.mpc(r)) for r in rts]
    xmin, xmax = float(min(rs)), float(max(rs))
    ymin, ymax = float(min(is_)), float(max(is_))
    
    # Expand bounds
    pad = 0.5
    width = max(xmax - xmin, 1)
    height_range = max(ymax - ymin, 1)
    xmin -= pad * width
    xmax += pad * width
    ymin -= pad * height_range
    ymax += pad * height_range
    
    bounds = [(xmin, xmax), (ymin, ymax)]
    
    # Global optimization using differential evolution
    result = differential_evolution(spread_np, bounds, seed=42, maxiter=2000, 
                                   tol=1e-15, atol=1e-20, workers=1, polish=True)
    
    # Convert to mpmath for high precision
    x_opt, y_opt = mp.mpf(str(result.x[0])), mp.mpf(str(result.x[1]))
    z = mp.mpc(x_opt, y_opt)
    
    # High precision refinement using gradient descent
    def spread_mp(z):
        us = saddle_points(a, b, c, z)
        hs = [height(a, b, c, z, u) for u in us]
        m = sum(hs)/3
        return sum((h-m)**2 for h in hs)
    
    # Newton-style refinement
    eps = mp.mpf(10)**(-40)
    for iteration in range(200):
        v0 = spread_mp(z)
        if v0 < mp.mpf(10)**(-60):
            break
            
        # Numerical gradient
        dz_re = (spread_mp(z + eps) - spread_mp(z - eps)) / (2*eps)
        dz_im = (spread_mp(z + eps*1j) - spread_mp(z - eps*1j)) / (2*eps)
        
        grad = mp.mpc(dz_re, dz_im)
        grad_norm = abs(grad)
        
        if grad_norm < mp.mpf(10)**(-60):
            break
        
        # Line search with backtracking
        step = mp.mpf(0.1)
        for _ in range(20):
            z_new = z - step * grad / grad_norm
            v_new = spread_mp(z_new)
            if v_new < v0:
                z = z_new
                break
            step /= 2
        else:
            break  # No improvement found
    
    return z

def branch_point_A(rts):
    a, b, c = coeffs_from_roots(rts)
    return (27*c - a**3) / (9*a**2 - 27*b)

# fixed 20-decimal formatting
def dec_fixed(x, places=20):
    getcontext().prec = places + 60
    s = mp.nstr(x, n=places+50)
    q = Decimal(1).scaleb(-places)
    d = Decimal(s).quantize(q, rounding=ROUND_HALF_UP)
    return format(d, "f")

def fmt20(z):
    re, im = mp.re(z), mp.im(z)
    sign = "+" if im >= 0 else "-"
    return f"{dec_fixed(re)} {sign} {dec_fixed(abs(im))}i"

v = vertex_degree3(roots)
A = branch_point_A(roots)

# Also compute and display the final spread/heights for verification
a, b, c = coeffs_from_roots(roots)
us = saddle_points(a, b, c, v)
hs = [height(a, b, c, v, u) for u in us]
spread = sum((h - sum(hs)/3)**2 for h in hs)

print("degree-3 vertex z_* (triple saddle tie):", fmt20(v))
print("branch point A (square in Fig. 9):     ", fmt20(A))
print()
print("Verification at z_*:")
print(f"  Heights: {[float(h) for h in hs]}")
print(f"  Spread (should be ~0): {float(spread):.2e}")

if float(spread) > 1e-10:
    print()
    print("WARNING: Spread is not close to zero!")
    print("  This set of roots may not have a true degree-3 vertex (triple saddle tie).")
    print("  The optimal z_* found minimizes the height variance but doesn't achieve")
    print("  exact equality of all three heights.")

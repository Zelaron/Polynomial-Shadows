"""
Newton Flow Wallpaper Generator
===============================
Generate beautiful 4K (3840x2160) visualizations of Newton's method flow
for polynomial and rational functions.

USER INSTRUCTIONS:
------------------
Simply edit the NUMERATOR_ROOTS and DENOMINATOR_ROOTS lists below.
- For a polynomial: set DENOMINATOR_ROOTS = [] or None
- For a rational function: provide both numerator and denominator roots

The code will automatically:
- Compute the derivative symbolically
- Find critical points analytically
- Determine a suitable viewing window
- Generate the flow visualization
"""

import numpy as np
import matplotlib.pyplot as plt

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              USER INPUT SECTION                              ║
# ║                                                                              ║
# ║  Define your function R(z) = P(z) / Q(z) by specifying roots.                ║
# ║  For a polynomial, leave DENOMINATOR_ROOTS empty.                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Numerator roots (zeros of R) - these are the attractors in Newton's method
NUMERATOR_ROOTS = [19+3j, 17-4j, -18-12j]

# Denominator roots (poles of R) - leave empty [] for a polynomial
# Example polynomial: DENOMINATOR_ROOTS = []
# Example rational:   DENOMINATOR_ROOTS = [9-6j, 5+6j]
DENOMINATOR_ROOTS = [9-6j, 5+6j]

# Optional leading coefficients (usually just leave as 1)
NUMERATOR_COEFF = 1
DENOMINATOR_COEFF = 1

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                            END USER INPUT SECTION                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# ==============================================================================
#                          POLYNOMIAL ARITHMETIC
# ==============================================================================

def poly_from_roots(roots, leading_coeff=1):
    """
    Build polynomial coefficients from roots.
    Returns coefficients [c_n, c_{n-1}, ..., c_1, c_0] (highest degree first).
    """
    if not roots:
        return np.array([leading_coeff], dtype=np.complex128)
    
    # Start with (z - r_0)
    coeffs = np.array([1, -roots[0]], dtype=np.complex128)
    
    # Multiply by (z - r_i) for each additional root
    for r in roots[1:]:
        new_coeffs = np.zeros(len(coeffs) + 1, dtype=np.complex128)
        new_coeffs[:-1] += coeffs        # z * previous
        new_coeffs[1:] -= r * coeffs     # -r * previous
        coeffs = new_coeffs
    
    return leading_coeff * coeffs


def poly_derivative_coeffs(coeffs):
    """
    Compute derivative of polynomial given coefficients.
    Input: [c_n, c_{n-1}, ..., c_1, c_0] (highest degree first)
    Output: derivative coefficients
    """
    n = len(coeffs) - 1
    if n <= 0:
        return np.array([0], dtype=np.complex128)
    
    degrees = np.arange(n, 0, -1)
    return coeffs[:-1] * degrees


def poly_multiply(p1, p2):
    """Multiply two polynomials given as coefficient arrays (highest degree first)."""
    return np.convolve(p1, p2)


def poly_subtract(p1, p2):
    """Subtract p2 from p1, handling different lengths."""
    # Pad shorter polynomial with leading zeros
    len1, len2 = len(p1), len(p2)
    if len1 > len2:
        p2 = np.concatenate([np.zeros(len1 - len2, dtype=np.complex128), p2])
    elif len2 > len1:
        p1 = np.concatenate([np.zeros(len2 - len1, dtype=np.complex128), p1])
    return p1 - p2


def eval_poly(coeffs, z):
    """Evaluate polynomial with given coefficients at points z (Horner's method)."""
    result = np.zeros_like(z, dtype=np.complex128)
    for c in coeffs:
        result = result * z + c
    return result


def poly_roots(coeffs):
    """Find roots of polynomial. Returns empty array for constant polynomials."""
    # Remove leading zeros
    coeffs = np.trim_zeros(coeffs, 'f')
    if len(coeffs) <= 1:
        return np.array([], dtype=np.complex128)
    return np.roots(coeffs)


# ==============================================================================
#                          RATIONAL FUNCTION CLASS
# ==============================================================================

class RationalFunction:
    """
    Represents a rational function R(z) = P(z) / Q(z) with automatic differentiation.
    """
    
    def __init__(self, num_roots, den_roots, num_coeff=1, den_coeff=1):
        self.num_roots = list(num_roots) if num_roots else []
        self.den_roots = list(den_roots) if den_roots else []
        self.num_coeff = num_coeff
        self.den_coeff = den_coeff
        
        # Build polynomial coefficients
        self.P_coeffs = poly_from_roots(self.num_roots, num_coeff)
        self.Q_coeffs = poly_from_roots(self.den_roots, den_coeff)
        
        # Compute derivative coefficients
        self.dP_coeffs = poly_derivative_coeffs(self.P_coeffs)
        self.dQ_coeffs = poly_derivative_coeffs(self.Q_coeffs)
        
        self.is_polynomial = len(self.den_roots) == 0
        
        # Compute critical points analytically
        self._critical_points = self._compute_critical_points()
    
    def __call__(self, z):
        """Evaluate R(z) = P(z) / Q(z)."""
        P = eval_poly(self.P_coeffs, z)
        Q = eval_poly(self.Q_coeffs, z)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            return P / Q
    
    def derivative(self, z):
        """
        Evaluate R'(z) using quotient rule:
        R'(z) = (P'(z)Q(z) - P(z)Q'(z)) / Q(z)^2
        
        For polynomial (Q=const): R'(z) = P'(z) / Q
        """
        P = eval_poly(self.P_coeffs, z)
        Q = eval_poly(self.Q_coeffs, z)
        dP = eval_poly(self.dP_coeffs, z)
        dQ = eval_poly(self.dQ_coeffs, z)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            return (dP * Q - P * dQ) / (Q * Q)
    
    def _compute_critical_points(self):
        """
        Find critical points where R'(z) = 0.
        
        R'(z) = (P'Q - PQ') / Q^2
        
        Critical points are roots of the numerator: P'Q - PQ' = 0
        (excluding any points that are also poles)
        """
        # Numerator of R'(z): P'(z)Q(z) - P(z)Q'(z)
        term1 = poly_multiply(self.dP_coeffs, self.Q_coeffs)
        term2 = poly_multiply(self.P_coeffs, self.dQ_coeffs)
        numerator_of_derivative = poly_subtract(term1, term2)
        
        # Find roots
        candidates = poly_roots(numerator_of_derivative)
        
        if len(candidates) == 0:
            return np.array([], dtype=np.complex128)
        
        # Filter out points that are too close to poles (those aren't true critical points)
        poles = np.array(self.den_roots, dtype=np.complex128) if self.den_roots else np.array([])
        
        valid_critical = []
        for cp in candidates:
            # Check if this is near a pole
            if len(poles) > 0:
                min_dist_to_pole = np.min(np.abs(cp - poles))
                if min_dist_to_pole < 1e-6:
                    continue  # Skip, this is at a pole
            
            # Verify it's actually a critical point (derivative ≈ 0)
            z_test = np.array([cp], dtype=np.complex128)
            deriv_val = self.derivative(z_test)[0]
            if np.abs(deriv_val) < 1e-6:
                valid_critical.append(cp)
        
        return np.array(valid_critical, dtype=np.complex128)
    
    def get_zeros(self):
        """Return the zeros (roots of numerator)."""
        return np.array(self.num_roots, dtype=np.complex128)
    
    def get_poles(self):
        """Return the poles (roots of denominator)."""
        return np.array(self.den_roots, dtype=np.complex128) if self.den_roots else np.array([])
    
    def get_critical_points(self):
        """Return the critical points (where R'(z) = 0)."""
        return self._critical_points


# ==============================================================================
#                         AUTOMATIC BOUNDING BOX
# ==============================================================================

def compute_bounding_box(zeros, poles, critical_points, aspect_ratio=16/9, padding_factor=0.25):
    """
    Compute a suitable bounding box containing all zeros, poles, and critical points
    with padding, adjusted to the target aspect ratio.
    """
    # Collect all significant points
    all_points = list(zeros)
    if len(poles) > 0:
        all_points.extend(list(poles))
    if len(critical_points) > 0:
        all_points.extend(list(critical_points))
    
    if not all_points:
        # Default box if no points
        return -10, 10, -10 * 9/16, 10 * 9/16
    
    all_points = np.array(all_points)
    
    x_coords = all_points.real
    y_coords = all_points.imag
    
    x_min_data, x_max_data = x_coords.min(), x_coords.max()
    y_min_data, y_max_data = y_coords.min(), y_coords.max()
    
    # Handle degenerate cases (all points on a line)
    x_range = max(x_max_data - x_min_data, 1.0)
    y_range = max(y_max_data - y_min_data, 1.0)
    
    # Add padding
    x_min = x_min_data - padding_factor * x_range
    x_max = x_max_data + padding_factor * x_range
    y_min = y_min_data - padding_factor * y_range
    y_max = y_max_data + padding_factor * y_range
    
    # Adjust to target aspect ratio (16:9 for 4K)
    current_width = x_max - x_min
    current_height = y_max - y_min
    current_ratio = current_width / current_height
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    if current_ratio < aspect_ratio:
        # Need to widen horizontally
        new_width = current_height * aspect_ratio
        x_min = x_center - new_width / 2
        x_max = x_center + new_width / 2
    else:
        # Need to heighten vertically
        new_height = current_width / aspect_ratio
        y_min = y_center - new_height / 2
        y_max = y_center + new_height / 2
    
    return x_min, x_max, y_min, y_max


# ==============================================================================
#                           FLOW SIMULATION
# ==============================================================================

def first_step(R, z, eps=1e-12, max_step=np.inf):
    """
    Compute the Newton step: -R(z) / R'(z)
    This gives the direction particles flow toward roots.
    """
    Rz = R(z)
    dRz = R.derivative(z)
    
    step = np.zeros_like(z, dtype=np.complex128)
    
    good = np.abs(dRz) > eps
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        step[good] = Rz[good] / dRz[good]
    
    if np.isfinite(max_step):
        mag = np.abs(step)
        too_big = mag > max_step
        step[too_big] *= (max_step / (mag[too_big] + 1e-30))
    
    return step


try:
    from scipy.ndimage import gaussian_filter as _gaussian_filter
    def gaussian_blur(img, sigma):
        return _gaussian_filter(img, sigma=sigma, mode="nearest")
except ImportError:
    def gaussian_blur(img, sigma):
        ny, nx = img.shape
        fy = np.fft.fftfreq(ny)[:, None]
        fx = np.fft.fftfreq(nx)[None, :]
        H = np.exp(-2.0 * (np.pi * sigma)**2 * (fx*fx + fy*fy))
        return np.fft.ifft2(np.fft.fft2(img) * H).real


# ==============================================================================
#                          4K WALLPAPER SETTINGS
# ==============================================================================

# Target: 3840 x 2160 (16:9 aspect ratio)
RES_BINS_X = 3840
RES_BINS_Y = 2160
ASPECT_RATIO = RES_BINS_X / RES_BINS_Y

# Simulation parameters
RES_PARTICLES = 4000          # Base particle resolution
TIME_STEPS = 800              # Number of flow steps

# Flow parameters
DEGREE = 3.5                  # Flow trail length multiplier
MAX_STEP = 5.0                # Maximum step size
TIME_DECAY = 1.0              # Trail fade rate

# Post-processing
CLIP_LO = 0.1
CLIP_HI = 99.9
UNSHARP_SIGMA = 1.2
UNSHARP_AMOUNT = 2.0
EDGE_GAIN = 0.15
FINAL_GAMMA = 0.82

# Output
OUTPUT_FILE = "newton_flow_4k_wallpaper.png"


# ==============================================================================
#                              MAIN EXECUTION
# ==============================================================================

def main():
    print("=" * 70)
    print("Newton Flow 4K Wallpaper Generator")
    print("=" * 70)
    
    # Build the rational function
    R = RationalFunction(NUMERATOR_ROOTS, DENOMINATOR_ROOTS, 
                         NUMERATOR_COEFF, DENOMINATOR_COEFF)
    
    zeros = R.get_zeros()
    poles = R.get_poles()
    critical_points = R.get_critical_points()
    
    func_type = "Polynomial" if R.is_polynomial else "Rational Function"
    print(f"\nFunction type: {func_type}")
    print(f"Degree of numerator: {len(NUMERATOR_ROOTS)}")
    if not R.is_polynomial:
        print(f"Degree of denominator: {len(DENOMINATOR_ROOTS)}")
    
    print(f"\nZeros (roots): {zeros}")
    if not R.is_polynomial:
        print(f"Poles: {poles}")
    print(f"Critical points: {critical_points}")
    print(f"Number of critical points found: {len(critical_points)}")
    
    # Compute bounding box
    x_min, x_max, y_min, y_max = compute_bounding_box(zeros, poles, critical_points, ASPECT_RATIO)
    print(f"\nAuto-computed bounding box:")
    print(f"  x ∈ [{x_min:.2f}, {x_max:.2f}]")
    print(f"  y ∈ [{y_min:.2f}, {y_max:.2f}]")
    
    # Initialize particles
    # Scale particle count to maintain density across aspect ratio
    n_particles_x = int(RES_PARTICLES * np.sqrt(ASPECT_RATIO))
    n_particles_y = int(RES_PARTICLES / np.sqrt(ASPECT_RATIO))
    
    print(f"\nInitializing {n_particles_x * n_particles_y:,} particles...")
    
    x = np.linspace(x_min, x_max, n_particles_x, dtype=np.float32)
    y = np.linspace(y_min, y_max, n_particles_y, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    Z0 = (X + 1j * Y).astype(np.complex128)
    
    print("Computing velocity field...")
    Velocity = (-first_step(R, Z0, eps=1e-12, max_step=MAX_STEP)).astype(np.complex64)
    
    # Histogram bins matching 4K output
    xbins = np.linspace(x_min, x_max, RES_BINS_X + 1, dtype=np.float32)
    ybins = np.linspace(y_min, y_max, RES_BINS_Y + 1, dtype=np.float32)
    
    combined = np.zeros((RES_BINS_Y, RES_BINS_X), dtype=np.float32)
    
    print(f"Simulating {TIME_STEPS} time steps...")
    
    u = np.linspace(0.0, 1.0, TIME_STEPS, dtype=np.float32)
    t_values = DEGREE * (u ** 2)
    
    Z0_f32 = Z0.astype(np.complex64)
    
    for i, t in enumerate(t_values):
        if (i + 1) % 100 == 0:
            print(f"  Step {i+1}/{TIME_STEPS}...")
        
        Zt = Z0_f32 + t * Velocity
        
        mask = ((Zt.real >= x_min) & (Zt.real <= x_max) & 
                (Zt.imag >= y_min) & (Zt.imag <= y_max))
        valid = Zt[mask]
        if valid.size == 0:
            continue
        
        w = np.exp(-TIME_DECAY * (t / DEGREE))
        
        H, _, _ = np.histogram2d(valid.imag, valid.real, bins=[ybins, xbins])
        combined += (w * H).astype(np.float32)
    
    print("Applying post-processing...")
    
    # Log transform for dynamic range
    img = np.log1p(combined)
    
    # Percentile normalization
    lo = np.percentile(img, CLIP_LO)
    hi = np.percentile(img, CLIP_HI)
    img = np.clip((img - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    
    # Unsharp masking for detail enhancement
    blur = gaussian_blur(img, sigma=UNSHARP_SIGMA)
    img = np.clip(img + UNSHARP_AMOUNT * (img - blur), 0.0, 1.0)
    
    # Subtle edge enhancement
    gy, gx = np.gradient(img)
    edges = np.hypot(gx, gy)
    edges /= (edges.max() + 1e-12)
    img = np.clip(img + EDGE_GAIN * edges, 0.0, 1.0)
    
    # Final gamma correction
    img = img ** FINAL_GAMMA
    
    # ==============================================================================
    #                              RENDERING
    # ==============================================================================
    
    print("Rendering 4K image...")
    
    # Figure size for exact 4K at 200 DPI: 3840/200 x 2160/200 = 19.2" x 10.8"
    fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor="black", dpi=200)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    ax.imshow(img, extent=[x_min, x_max, y_min, y_max], origin="lower",
              cmap="inferno", interpolation="lanczos", vmin=0, vmax=1, aspect='auto')
    
    # Plot zeros (always)
    ax.scatter(zeros.real, zeros.imag, c="cyan", s=120, marker="o",
               edgecolors="white", linewidth=1.5, label="Zeros", zorder=10)
    
    # Plot poles (only for rational functions)
    if not R.is_polynomial and len(poles) > 0:
        ax.scatter(poles.real, poles.imag, c="red", s=120, marker="s",
                   edgecolors="white", linewidth=1.5, label="Poles", zorder=10)
    
    # Plot critical points (always, if any exist)
    if len(critical_points) > 0:
        ax.scatter(critical_points.real, critical_points.imag, c="lime", s=120, marker="x",
                   linewidth=2.5, label="Critical Points", zorder=10)
    
    ax.legend(loc="upper right", framealpha=0.3, labelcolor="white", fontsize=14)
    ax.axis("off")
    
    plt.savefig(OUTPUT_FILE, dpi=200, facecolor="black", 
                bbox_inches='tight', pad_inches=0)
    plt.show()
    
    print(f"\n{'=' * 70}")
    print(f"Saved 4K wallpaper to: {OUTPUT_FILE}")
    print(f"Dimensions: {RES_BINS_X} x {RES_BINS_Y} pixels")
    print("=" * 70)


if __name__ == "__main__":
    main()

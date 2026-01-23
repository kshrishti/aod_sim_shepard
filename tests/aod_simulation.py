"""
AOD Beam Profile Simulation with Phase Discontinuity Propagation

This module simulates the electric field in the focal plane of a lens after
an Acousto-Optic Deflector (AOD) when a phase discontinuity propagates through
the beam due to a phase reset in the RF drive signal.

Physical setup:
- RF continuously chirps from f0 to f1
- At t=0, phase is reset (creating discontinuity Δφ)
- Discontinuity propagates through crystal at acoustic velocity V
- Lens Fourier-transforms the beam to focal plane
- We simulate during the transit time (discontinuity crossing the beam)
"""

import numpy as np
from scipy import integrate
from dataclasses import dataclass
from typing import Tuple, Callable
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm


@dataclass
class AODParameters:
    """Physical parameters for the AOD system"""
    
    # RF parameters
    f0: float = 50e6          # Start frequency [Hz]
    f1: float = 80e6          # End frequency [Hz]
    alpha: float = 1e12       # Chirp rate [Hz/s] (1 MHz/μs)
    phi0: float = 0.0         # Phase offset [rad]
    
    # Acoustic parameters
    V: float = 650.0          # Acoustic velocity [m/s] (TeO2)
    
    # Optical parameters
    wavelength: float = 780e-9  # Wavelength [m]
    w0: float = 300e-6        # Beam waist [m]
    F: float = 200e-3         # Focal length [m]
    M: float = 1.0            # Magnification before AOD

    # Crystal geometry
    crystal_length: float = 7.5e-3  # Crystal length [m]
    beam_position: float = 3.75e-3  # Beam center position from transducer [m]
    
    # Crystal parameters
    @property
    def k(self) -> float:
        """Wave number [rad/m]"""
        return 2 * np.pi / self.wavelength

    @property
    def T(self) -> float:
        """Transit time for discontinuity to cross beam [s]"""
        return self.w0 * self.M / self.V

    @property
    def delta_f(self) -> float:
        """Chirp range [Hz]"""
        return self.f1 - self.f0

    @property
    def chirp_period(self) -> float:
        """Period of one chirp cycle [s]"""
        return self.delta_f / self.alpha

    @property
    def phi0_for_continuity(self) -> float:
        """Phase offset to ensure continuity at frequency reset [rad]

        When frequency resets from f1 back to f0, the accumulated phase
        over one chirp period is:
        φ₀ = ∫₀^T_chirp 2πf(t)dt = 2π[f₀·T + (α/2)·T²]
           = π(f₁² - f₀²)/α
        """
        T_chirp = self.chirp_period
        return 2 * np.pi * (self.f0 * T_chirp + 0.5 * self.alpha * T_chirp**2)
    
    def __post_init__(self):
        """Validate parameters"""
        assert self.f1 > self.f0, "f1 must be greater than f0"
        assert self.alpha > 0, "Chirp rate must be positive"
        assert self.V > 0, "Acoustic velocity must be positive"
        assert self.w0 > 0, "Beam waist must be positive"


class AODSimulation:
    """Simulate AOD beam profile with propagating phase discontinuity"""
    
    def __init__(self, params: AODParameters):
        self.params = params
        
    def calculate_phase(self, xi: np.ndarray, eta: np.ndarray, t: float,
                       use_frequency_reset: bool = True) -> np.ndarray:
        """
        Calculate phase at position (xi, eta) and time t

        Models a frequency reset from f₁ → f₀ that propagates through the beam.
        Different spatial positions sample different parts of the RF waveform due
        to acoustic propagation delay:
        - OLD region (xi ≥ xi_disc): Pre-reset RF, at high frequencies (near f₁)
        - NEW region (xi < xi_disc): Post-reset RF, at low frequencies (near f₀)

        Phase continuity is maintained by adding φ₀ to the old region.

        Args:
            xi: Transverse position(s) in crystal [m]
            eta: Transverse position(s) in crystal [m]
            t: Time [s]
            use_frequency_reset: If True, uses frequency reset model with φ₀
                                If False, regions have same phase (no discontinuity)

        Returns:
            Phase array [rad]
        """
        p = self.params

        # Time delay for acoustic wave propagation
        # Positive xi means the wave reaches that point later
        tau = t - p.T/2 - xi * p.M / p.V

        # Position of discontinuity wavefront at time t
        # At t=0, reset occurs and discontinuity is at xi = -p.w0/2 (left edge)
        # It propagates rightward at velocity V/M
        xi_disc = -p.w0/2 + p.V * t / p.M

        if use_frequency_reset:
            # At t=0, the frequency resets from f₁ → f₀ at the transducer
            # OLD region (xi >= xi_disc): Has NOT seen reset, still at high freq near f₁
            # NEW region (xi < xi_disc): HAS seen reset, at low freq near f₀

            # NEW region: Just reset to f₀, so tau starts from when reset occurred
            tau_new = tau
            phi_new = 2 * np.pi * (p.f0 * tau_new + 0.5 * p.alpha * tau_new**2) + p.phi0

            # OLD region: Was chirping before reset, add offset to be near f₁
            # Time to chirp from f₀ to f₁
            tau_old = tau + p.chirp_period
            phi_old = 2 * np.pi * (p.f0 * tau_old + 0.5 * p.alpha * tau_old**2) + p.phi0

            # Apply the appropriate phase based on position relative to discontinuity
            phase = np.where(xi >= xi_disc, phi_old, phi_new)
        else:
            # No frequency reset - uniform phase (for testing/comparison)
            phi_base = 2 * np.pi * (p.f0 * tau + 0.5 * p.alpha * tau**2) + p.phi0
            phase = phi_base

        return phase
    
    def integrand(self, xi: float, eta: float, u: float, v: float,
                  t: float) -> complex:
        """
        Evaluate the integrand for the electric field integral

        E_out ∝ ∫∫ exp(-jφ) × exp(-(ξ²+η²)M²/w₀²) ×
                exp(-jk(ξ²+η²)/(2F)) × exp(-j2π(uξ+vη)/(λF)) dξdη

        Args:
            xi, eta: Integration variables (position in crystal) [m]
            u, v: Focal plane coordinates [m]
            t: Time [s]

        Returns:
            Complex integrand value
        """
        p = self.params

        # Phase from AOD (with frequency reset propagating through beam)
        phi = self.calculate_phase(xi, eta, t)
        
        # Gaussian beam profile
        gaussian = np.exp(-(xi**2 + eta**2) * p.M**2 / p.w0**2)
        
        # Quadratic phase (propagation to lens)
        quad_phase = np.exp(-1j * p.k * (xi**2 + eta**2) / (2 * p.F))
        
        # Fourier transform kernel (lens + propagation to focal plane)
        fourier_kernel = np.exp(-1j * 2 * np.pi * (u * xi + v * eta) / 
                               (p.wavelength * p.F))
        
        # AOD phase modulation
        aod_phase = np.exp(-1j * phi)
        
        return aod_phase * gaussian * quad_phase * fourier_kernel
    
    def compute_field_point(self, u: float, v: float, t: float,
                           integration_method: str = 'quad') -> complex:
        """
        Compute electric field at a single point (u, v) in focal plane at time t

        Args:
            u, v: Focal plane coordinates [m]
            t: Time [s]
            integration_method: 'quad' for adaptive quadrature

        Returns:
            Complex electric field value
        """
        p = self.params

        # Integration limits: ±3 beam waists (captures >99% of energy)
        xi_max = 3 * p.w0
        eta_max = 3 * p.w0

        if integration_method == 'quad':
            # Adaptive 2D integration (slower but accurate)
            result_real, _ = integrate.dblquad(
                lambda eta, xi: np.real(self.integrand(xi, eta, u, v, t)),
                -xi_max, xi_max,
                -eta_max, eta_max,
                epsabs=1e-6, epsrel=1e-6
            )

            result_imag, _ = integrate.dblquad(
                lambda eta, xi: np.imag(self.integrand(xi, eta, u, v, t)),
                -xi_max, xi_max,
                -eta_max, eta_max,
                epsabs=1e-6, epsrel=1e-6
            )

            return result_real + 1j * result_imag

        else:
            raise ValueError(f"Unknown integration method: {integration_method}")
    
    def compute_field_grid(self, u_array: np.ndarray, v_array: np.ndarray,
                          t: float,
                          n_points: int = 200) -> np.ndarray:
        """
        Compute electric field on a grid in the focal plane using FFT method

        This is much faster than point-by-point integration and suitable for
        visualization. Uses discrete Fourier transform relationship.

        Args:
            u_array: 1D array of u coordinates [m]
            v_array: 1D array of v coordinates [m]
            t: Time [s]
            n_points: Number of points for integration grid

        Returns:
            2D array of complex electric field values
        """
        p = self.params

        # Create integration grid in (xi, eta) space
        xi_max = 3 * p.w0
        eta_max = 3 * p.w0

        xi = np.linspace(-xi_max, xi_max, n_points)
        eta = np.linspace(-eta_max, eta_max, n_points)
        XI, ETA = np.meshgrid(xi, eta, indexing='ij')

        # Calculate phase on the grid (with frequency reset model)
        phi = self.calculate_phase(XI, ETA, t)
        
        # Calculate the field in the aperture
        gaussian = np.exp(-(XI**2 + ETA**2) * p.M**2 / p.w0**2)
        quad_phase = np.exp(-1j * p.k * (XI**2 + ETA**2) / (2 * p.F))
        aod_phase = np.exp(-1j * phi)
        
        aperture_field = aod_phase * gaussian * quad_phase
        
        # Use FFT to compute Fourier transform
        # The Fourier transform gives us the field in the focal plane
        field_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(aperture_field)))
        
        # Create frequency arrays corresponding to focal plane coordinates
        d_xi = xi[1] - xi[0]
        d_eta = eta[1] - eta[0]
        
        freq_xi = np.fft.fftshift(np.fft.fftfreq(n_points, d_xi))
        freq_eta = np.fft.fftshift(np.fft.fftfreq(n_points, d_eta))
        
        # Convert frequency to focal plane coordinates: u = λF·freq
        u_fft = p.wavelength * p.F * freq_xi
        v_fft = p.wavelength * p.F * freq_eta
        
        # Interpolate to desired output grid
        from scipy.interpolate import RectBivariateSpline
        
        interp_real = RectBivariateSpline(u_fft, v_fft, np.real(field_fft))
        interp_imag = RectBivariateSpline(u_fft, v_fft, np.imag(field_fft))
        
        U, V = np.meshgrid(u_array, v_array, indexing='ij')
        
        field_real = interp_real.ev(U.ravel(), V.ravel()).reshape(U.shape)
        field_imag = interp_imag.ev(U.ravel(), V.ravel()).reshape(U.shape)
        
        # Apply proper scaling
        scale = d_xi * d_eta
        
        return scale * (field_real + 1j * field_imag)
    
    def simulate_time_series(self, n_time_steps: int,
                            u_range: Tuple[float, float],
                            v_range: Tuple[float, float],
                            n_spatial_points: int = 200,
                            n_integration_points: int = 200) -> dict:
        """
        Simulate the beam profile evolution as frequency reset propagates

        Models RF frequency resetting from f₁ → f₀. The reset propagates through
        the crystal at acoustic velocity, creating spatial regions with different
        frequency content that affect the focal plane beam profile.

        Args:
            n_time_steps: Number of time steps during transit
            u_range: (u_min, u_max) for focal plane [m]
            v_range: (v_min, v_max) for focal plane [m]
            n_spatial_points: Grid resolution for output
            n_integration_points: Grid resolution for integration

        Returns:
            Dictionary containing:
                - 'time': Time array [s]
                - 'u': u coordinate array [m]
                - 'v': v coordinate array [m]
                - 'field': 3D array (time, u, v) of complex field
                - 'intensity': 3D array (time, u, v) of intensity
                - 'phase': 3D array (time, u, v) of phase [rad]
        """
        p = self.params

        # Time array: from t=0 (reset enters) to t=T (reset exits beam)
        time = np.linspace(0, p.T, n_time_steps)

        # Focal plane coordinate arrays
        u = np.linspace(u_range[0], u_range[1], n_spatial_points)
        v = np.linspace(v_range[0], v_range[1], n_spatial_points)

        # Preallocate arrays
        field = np.zeros((n_time_steps, n_spatial_points, n_spatial_points),
                        dtype=complex)

        print(f"Simulating {n_time_steps} time steps...")
        print(f"Transit time: {p.T*1e6:.3f} μs")
        print(f"Chirp rate: {p.alpha/1e12:.3f} MHz/μs")
        print(f"Frequency range: {p.f0/1e6:.1f} - {p.f1/1e6:.1f} MHz")
        print(f"Phase offset for continuity: {p.phi0_for_continuity:.3f} rad = {p.phi0_for_continuity/(2*np.pi):.3f} cycles")

        # Compute field at each time step
        for i, t in enumerate(time):
            print(f"  Computing time step {i+1}/{n_time_steps} (t = {t*1e6:.3f} μs)...",
                  end='\r')

            field[i] = self.compute_field_grid(u, v, t,
                                              n_points=n_integration_points)
        
        print("\nSimulation complete!")
        
        # Calculate intensity and phase
        intensity = np.abs(field)**2
        phase = np.angle(field)
        
        return {
            'time': time,
            'u': u,
            'v': v,
            'field': field,
            'intensity': intensity,
            'phase': phase,
            'params': p
        }


def plot_time_snapshot(results: dict, time_index: int, 
                      save_path: str = None, show_phase: bool = True):
    """
    Plot intensity and phase at a specific time
    
    Args:
        results: Dictionary from simulate_time_series
        time_index: Index of time step to plot
        save_path: If provided, save figure to this path
        show_phase: Whether to show phase plot
    """
    u = results['u'] * 1e3  # Convert to mm
    v = results['v'] * 1e3
    intensity = results['intensity'][time_index]
    phase = results['phase'][time_index]
    t = results['time'][time_index] * 1e6  # Convert to μs
    T = results['params'].T * 1e6
    
    if show_phase:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    # Intensity plot
    im1 = ax1.imshow(intensity.T, extent=[u[0], u[-1], v[0], v[-1]],
                     origin='lower', aspect='auto', cmap='hot',
                     norm=PowerNorm(gamma=0.5))
    ax1.set_xlabel('u [mm]')
    ax1.set_ylabel('v [mm]')
    ax1.set_title(f'Intensity at t = {t:.3f} μs (t/T = {t/T:.2f})')
    plt.colorbar(im1, ax=ax1, label='|E|²')
    
    if show_phase:
        # Phase plot
        im2 = ax2.imshow(phase.T, extent=[u[0], u[-1], v[0], v[-1]],
                        origin='lower', aspect='auto', cmap='twilight',
                        vmin=-np.pi, vmax=np.pi)
        ax2.set_xlabel('u [mm]')
        ax2.set_ylabel('v [mm]')
        ax2.set_title(f'Phase at t = {t:.3f} μs')
        plt.colorbar(im2, ax=ax2, label='arg(E) [rad]')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cross_sections(results: dict, time_index: int, save_path: str = None):
    """
    Plot cross-sections through the center of the beam
    
    Args:
        results: Dictionary from simulate_time_series
        time_index: Index of time step to plot
        save_path: If provided, save figure to this path
    """
    u = results['u'] * 1e3  # Convert to mm
    v = results['v'] * 1e3
    intensity = results['intensity'][time_index]
    t = results['time'][time_index] * 1e6
    T = results['params'].T * 1e6
    
    # Find center indices
    center_u = len(u) // 2
    center_v = len(v) // 2
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Horizontal cross-section (v = 0)
    ax1.plot(u, intensity[:, center_v], 'b-', linewidth=2)
    ax1.set_xlabel('u [mm]')
    ax1.set_ylabel('Intensity |E|²')
    ax1.set_title(f'Horizontal cross-section (v=0) at t = {t:.3f} μs')
    ax1.grid(True, alpha=0.3)
    
    # Vertical cross-section (u = 0)
    ax2.plot(v, intensity[center_u, :], 'r-', linewidth=2)
    ax2.set_xlabel('v [mm]')
    ax2.set_ylabel('Intensity |E|²')
    ax2.set_title(f'Vertical cross-section (u=0) at t = {t:.3f} μs')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_beam_properties(results: dict, time_index: int) -> dict:
    """
    Calculate beam properties (centroid, width, peak intensity, etc.)
    
    Args:
        results: Dictionary from simulate_time_series
        time_index: Index of time step to analyze
        
    Returns:
        Dictionary of beam properties
    """
    u = results['u']
    v = results['v']
    intensity = results['intensity'][time_index]
    
    # Total power
    du = u[1] - u[0]
    dv = v[1] - v[0]
    total_power = np.sum(intensity) * du * dv
    
    # Centroid
    U, V = np.meshgrid(u, v, indexing='ij')
    u_centroid = np.sum(U * intensity) * du * dv / total_power
    v_centroid = np.sum(V * intensity) * du * dv / total_power
    
    # Second moments (beam width)
    u_width = np.sqrt(np.sum((U - u_centroid)**2 * intensity) * du * dv / total_power)
    v_width = np.sqrt(np.sum((V - v_centroid)**2 * intensity) * du * dv / total_power)
    
    # Peak intensity and location
    peak_intensity = np.max(intensity)
    peak_idx = np.unravel_index(np.argmax(intensity), intensity.shape)
    u_peak = u[peak_idx[0]]
    v_peak = v[peak_idx[1]]
    
    return {
        'total_power': total_power,
        'u_centroid': u_centroid,
        'v_centroid': v_centroid,
        'u_width': u_width,
        'v_width': v_width,
        'peak_intensity': peak_intensity,
        'u_peak': u_peak,
        'v_peak': v_peak
    }


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("AOD Phase Discontinuity Simulation")
    print("=" * 60)
    
    # Create parameters with default values
    params = AODParameters(
        f0=50e6,           # 50 MHz
        f1=80e6,           # 80 MHz
        alpha=1e12,        # 1 MHz/μs chirp rate
        phi0=0.0,          # No additional phase offset
        V=650.0,           # TeO2 acoustic velocity
        wavelength=780e-9, # 780 nm
        w0=300e-6,         # 300 μm beam waist
        F=200e-3,          # 200 mm focal length
        M=1.0              # No magnification
    )
    
    print(f"\nPhysical parameters:")
    print(f"  Frequency range: {params.f0/1e6:.1f} - {params.f1/1e6:.1f} MHz")
    print(f"  Chirp rate: {params.alpha/1e12:.3f} MHz/μs")
    print(f"  Chirp period: {params.chirp_period*1e6:.2f} μs")
    print(f"  Beam waist: {params.w0*1e6:.0f} μm")
    print(f"  Transit time: {params.T*1e6:.3f} μs")
    print(f"  Wavelength: {params.wavelength*1e9:.0f} nm")
    print(f"  Focal length: {params.F*1e3:.0f} mm")
    print(f"  Phase offset for continuity: {params.phi0_for_continuity:.3f} rad = {params.phi0_for_continuity/(2*np.pi):.1f} cycles")

    # Create simulation
    sim = AODSimulation(params)
    
    # Define focal plane region of interest
    # Estimate deflection angle: θ ≈ λf/(V)
    # Focal plane displacement: u ≈ F·θ
    u_max = params.F * params.wavelength * params.f1 / params.V
    print(f"\nEstimated max deflection: {u_max*1e3:.2f} mm")
    
    # Diffraction-limited spot size in focal plane: ~λF/w₀
    spot_size = params.wavelength * params.F / params.w0
    print(f"Diffraction-limited spot size: {spot_size*1e6:.1f} μm")
    
    # Zoom in: show ±5 spot sizes around the deflected beam
    u_center = -u_max  # Beam is deflected negative due to phase term
    v_center = 0
    fov_size = 10 * spot_size  # Field of view: 10 spot sizes
    
    u_range = (u_center - fov_size/2, u_center + fov_size/2)
    v_range = (v_center - fov_size/2, v_center + fov_size/2)
    
    print(f"Focal plane range (zoomed): u ∈ [{u_range[0]*1e3:.2f}, {u_range[1]*1e3:.2f}] mm")
    print(f"                            v ∈ [{v_range[0]*1e3:.2f}, {v_range[1]*1e3:.2f}] mm")
    print(f"Field of view: {fov_size*1e3:.3f} mm = {fov_size/spot_size:.1f} spot sizes")
    
    # Run simulation with higher resolution
    results = sim.simulate_time_series(
        n_time_steps=10,
        u_range=u_range,
        v_range=v_range,
        n_spatial_points=300,       # Increased from 150
        n_integration_points=400    # Increased from 200
    )
    
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)
    
    # Plot snapshots at different times
    time_indices = [0, len(results['time'])//4, len(results['time'])//2, 
                   3*len(results['time'])//4, -1]
    
    for i, idx in enumerate(time_indices):
        print(f"\nPlotting time step {idx}...")
        plot_time_snapshot(results, idx,
                          save_path=f'aod_snapshot_{i}.png')

        # Analyze beam properties
        props = analyze_beam_properties(results, idx)
        t = results['time'][idx] * 1e6
        print(f"  Time: {t:.3f} μs")
        print(f"  Peak intensity: {props['peak_intensity']:.2e}")
        print(f"  Peak at: u = {props['u_peak']*1e3:.3f} mm, v = {props['v_peak']*1e3:.3f} mm")
        print(f"  Centroid: u = {props['u_centroid']*1e3:.3f} mm, v = {props['v_centroid']*1e3:.3f} mm")
        print(f"  Beam width: u_rms = {props['u_width']*1e6:.1f} μm, v_rms = {props['v_width']*1e6:.1f} μm")

    # Plot cross-sections at middle time
    plot_cross_sections(results, len(results['time'])//2,
                       save_path='aod_cross_sections.png')
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)

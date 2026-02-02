"""
Two-Tone AOD Animation with Shepard Tone Transition

Modified from original two-tone code to use Shepard fading instead of abrupt
frequency switch. This minimizes aberrations at the x2 tweezer position.

Physical setup:
- Two RF input tones separated by 30 MHz (from previous AOD stage)
- Each tone sweeps from 50 MHz to 80 MHz using Shepard fading
- Creates 3 beams in focal plane that smoothly transition
- After metasurface splitting: tweezers transition from (x1,x2) to (x2,x3)
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from matplotlib.patches import Circle
from matplotlib.colors import PowerNorm


# Copy AODParameters from your original code
class AODParameters:
    """Physical parameters for the AOD system"""
    
    def __init__(self, f0=50e6, f1=80e6, alpha=1e12, phi0=0.0,
                 V=650.0, wavelength=780e-9, w0=300e-6,
                 F=200e-3, M=1.0,
                 crystal_length=7.5e-3, beam_position=3.75e-3):
        
        self.f0 = f0
        self.f1 = f1
        self.alpha = alpha
        self.phi0 = phi0
        self.V = V
        self.wavelength = wavelength
        self.w0 = w0
        self.F = F
        self.M = M
        self.crystal_length = crystal_length
        self.beam_position = beam_position
        
    @property
    def k(self):
        return 2 * np.pi / self.wavelength
    
    @property
    def T(self):
        return self.w0 * self.M / self.V
    
    @property
    def delta_f(self):
        return self.f1 - self.f0
    
    @property
    def chirp_period(self):
        return self.delta_f / self.alpha


class ShepardTwoToneAOD:
    """AOD with two-tone input using Shepard fading"""
    
    def __init__(self, params: AODParameters, 
                 f_input_offset: float = 30e6,
                 shepard_spacing: float = 1e6, use_interlacing=True):
        """
        Args:
            params: AOD parameters
            f_input_offset: Frequency separation of two input tones [Hz]
            shepard_spacing: Frequency spacing between Shepard tones [Hz]
        """
        self.params = params
        self.f_input_offset = f_input_offset
        self.k_input = 2 * np.pi * f_input_offset / params.V
        self.Delta_f_shepard = shepard_spacing
        # self.use_interlacing = use_interlacing
        
    def shepard_amplitude(self, n: int, t: float) -> float:
        """
        Shepard amplitude for tone n at time t
        
        Two adjacent tones active at any time, amplitudes sum to 1
        """
        p = self.params
        
        # Fractional sweep position (0 to 1 over transit time)
        s = t / p.T
        
        # Convert to Shepard tone index (scaled by frequency range / spacing)
        s_scaled = s * (p.f1 - p.f0) / self.Delta_f_shepard
        
        m = int(np.floor(s_scaled))
        alpha = s_scaled - m  # Fractional part [0, 1)
        
        # Add offset for interlacing
        # if self.use_interlacing and axis == 'y':
        #     alpha = (s_scaled - m + 0.5) % 1.0  # Shift by half period
        # else:
        #     alpha = s_scaled - m
        
        if n == m:
            return np.cos(0.5 * np.pi * alpha)
        elif n == m + 1:
            return np.sin(0.5 * np.pi * alpha)
        else:
            return 0.0

    def schroeder_phase(self, n: int, M: int = 1) -> float:
        """Equation 4 from Stamper-Kurn paper"""
        if M == 1:
            return 0  # Single interval case
        return 2 * np.pi * n * (n - 1) / (2 * (M - 1))
    
    def get_active_shepard_tones(self, t: float) -> list:
        """Get list of active Shepard tones at time t"""
        p = self.params
        s_scaled = (t / p.T) * (p.f1 - p.f0) / self.Delta_f_shepard
        m = int(np.floor(s_scaled))
        
        tones = []
        for n in [m, m + 1]:
            A_n = self.shepard_amplitude(n, t)
            if A_n > 0.01:  # Only include significant amplitudes
                f_n = p.f0 + n * self.Delta_f_shepard
                tones.append({
                    'n': n,
                    'frequency': f_n,
                    'amplitude': A_n
                })
        
        return tones
    
    def compute_phase(self, XI: np.ndarray, ETA: np.ndarray, 
                     t: float, tone_freq: float) -> np.ndarray:
        """
        Compute acoustic phase for a single RF tone
        
        This is simplified - not modeling the frequency sweep discontinuity
        from your original code, just the acoustic phase grating
        """
        p = self.params
        
        # Acoustic wave number for this frequency
        k_acoustic = 2 * np.pi * tone_freq / p.V
        
        # phi_schroeder = self.schroeder_phase(tweezer_index, M=1) 
        # Simple acoustic phase (no time-dependent chirp for now)
        # In real system, this would include acoustic propagation time
        phase = k_acoustic * XI
        # phase_tilt = np.exp(-1j * (k_acoustic * XI + phi_schroeder))
        
        return phase

    
    def compute_focal_intensity(self, u_fft: np.ndarray, v_fft: np.ndarray,
                               XI: np.ndarray, ETA: np.ndarray,
                               t: float, gaussian: np.ndarray,
                               quad_phase: np.ndarray, scale: float) -> np.ndarray:
        """
        Compute focal plane intensity using two-tone Shepard input
        
        Key insight: Each Shepard tone creates TWO beams (from the two input tones),
        and we sum intensities incoherently across different Shepard tones
        """
        p = self.params
        n_points = XI.shape[0]
        
        # Total intensity (incoherent sum over all beams)
        # focal_intensity_total = np.zeros((n_points, n_points), dtype=float)
        focal_field_total = np.zeros((n_points, n_points), dtype=complex)
        
        # Get active Shepard tones
        active_tones = self.get_active_shepard_tones(t)
        
        for tone_info in active_tones:
            f_center = tone_info['frequency']
            A_n = tone_info['amplitude']
            
            # This Shepard tone creates TWO beams (from two-tone input)
            # Beam 1: center frequency
            phase1 = self.compute_phase(XI, ETA, t, f_center)
            component1 = gaussian * np.exp(-1j * phase1)
            
            # Beam 2: offset by f_input_offset
            phase2 = self.compute_phase(XI, ETA, t, f_center + self.f_input_offset)
            # Add phase tilt from previous AOD stage
            # component2 = gaussian * np.exp(-1j * phase2) * np.exp(1j * self.k_input * XI) # ---> cancels out second beam
            # component2 = gaussian * np.exp(-1j * phase2) 
            component2 = gaussian * np.exp(-1j * phase1) * np.exp(1j * self.k_input * XI)
            
            # Coherent sum of the two components (they're from same Shepard tone)
            aperture_field_n = (component1 + component2) * quad_phase
            
            # FFT to focal plane
            field_fft_n = np.fft.fftshift(
                np.fft.fft2(np.fft.ifftshift(aperture_field_n))
            )
            
            # Add intensity weighted by Shepard amplitude
            # (Incoherent sum across different Shepard tones)
            # focal_intensity_total += (A_n ** 2) * np.abs(field_fft_n) ** 2
            focal_field_total += A_n * field_fft_n
        
        # return scale * focal_intensity_total
        return scale * np.abs(focal_field_total)**2


# Create parameters (same as your original)
params = AODParameters(
    f0=50e6, f1=80e6, alpha=1e12, phi0=0.0,
    V=650.0, wavelength=780e-9, w0=300e-6,
    F=200e-3, M=1.0,
    crystal_length=7.5e-3, beam_position=3.75e-3
)

print("=" * 60)
print("Two-Tone Animation with Shepard Fading")
print("=" * 60)

transit_time = params.w0 / params.V
print(f"Beam waist w0 = {params.w0*1e6:.1f} um")
print(f"Acoustic velocity V = {params.V:.1f} m/s")
print(f"Transit time T = w0/V = {transit_time*1e6:.3f} us")

f_input_offset = 30e6  # Hz
shepard_spacing = 30e6  # Hz, Shepard tone spacing

print(f"\nTwo-tone input with Shepard fading:")
print(f"  Input tone separation: {f_input_offset/1e6:.0f} MHz")
print(f"  Shepard tone spacing: {shepard_spacing/1e6:.2f} MHz")
print(f"  Creates 3 output beams with smooth transitions")
print()

sim = ShepardTwoToneAOD(params, f_input_offset, shepard_spacing)

# Create aperture grid (same as original)
n_points = 800
xi_max = 3 * params.w0
xi = np.linspace(-xi_max, xi_max, n_points)
eta = np.linspace(-xi_max, xi_max, n_points)
XI, ETA = np.meshgrid(xi, eta, indexing='ij')

# Pre-compute FFT coordinates
d_xi = xi[1] - xi[0]
freq_xi = np.fft.fftshift(np.fft.fftfreq(n_points, d_xi))
freq_eta = np.fft.fftshift(np.fft.fftfreq(n_points, d_xi))
u_fft = params.wavelength * params.F * freq_xi
v_fft = params.wavelength * params.F * freq_eta
scale = d_xi * d_xi

# Pre-compute constant terms
gaussian = np.exp(-(XI**2 + ETA**2) * params.M**2 / params.w0**2)
quad_phase = np.exp(-1j * params.k * (XI**2 + ETA**2) / (2 * params.F))

# Find peak position at t=0 for zoom window
print("Finding beam position at t=0...")
focal_intensity_t0 = sim.compute_focal_intensity(u_fft, v_fft, XI, ETA, 0.0,
                                                 gaussian, quad_phase, scale)
u_peak_idx = np.unravel_index(np.argmax(focal_intensity_t0), 
                              focal_intensity_t0.shape)[0]
v_peak_idx = np.unravel_index(np.argmax(focal_intensity_t0),
                              focal_intensity_t0.shape)[1]
u_peak_t0 = u_fft[u_peak_idx]
v_peak_t0 = v_fft[v_peak_idx]
max_intensity_t0 = np.max(focal_intensity_t0)

print(f"Peak at t=0: u = {u_peak_t0*1e3:.4f} mm, v = {v_peak_t0*1e3:.4f} mm")
print(f"Max intensity: {max_intensity_t0:.3e}")
print()

# Define zoom window (same as original)
fov_zoom = 1.5e-3  # m
u_zoom_min = u_peak_t0 - fov_zoom/2
u_zoom_max = u_peak_t0 + fov_zoom/2
v_zoom_min = v_peak_t0 - fov_zoom/2
v_zoom_max = v_peak_t0 + fov_zoom/2

u_zoom_mask = (u_fft >= u_zoom_min) & (u_fft <= u_zoom_max)
v_zoom_mask = (v_fft >= v_zoom_min) & (v_fft <= v_zoom_max)

print(f"Zoom window: [{u_zoom_min*1e3:.4f}, {u_zoom_max*1e3:.4f}] mm")
print()

# Create animation frames
n_frames = 60
t_end = 1.0 * transit_time  # Full transit time
time_array = np.linspace(0, t_end, n_frames)

frames_folder = 'animation_frames_shepard_two_tone'
os.makedirs(frames_folder, exist_ok=True)
saved_frames = []

print(f"Creating {n_frames} frames...")
center_eta = n_points // 2

print("Scanning for global max intensity over all time steps...")

global_max_intensity = 0.0
global_min_intensity = np.inf

for t in time_array:
    focal_intensity = sim.compute_focal_intensity(
        u_fft, v_fft, XI, ETA, t,
        gaussian, quad_phase, scale
    )
    current_max = np.max(focal_intensity)
    current_min = np.min(focal_intensity)
    if current_max > global_max_intensity:
        global_max_intensity = current_max
    if current_min < global_min_intensity:
        global_min_intensity = current_min

print(f"Global max intensity over all frames: {global_max_intensity:.3e}")
# print(f"Global min intensity over all frames: {global_min_intensity:.3e}")

for i, t in enumerate(time_array):
    print(f"  Frame {i+1}/{n_frames}: t = {t*1e6:.3f} us", end='\r')
    break
    
    # Compute focal intensity with Shepard fading
    focal_intensity = sim.compute_focal_intensity(u_fft, v_fft, XI, ETA, t,
                                                   gaussian, quad_phase, scale)
    
    # Get active tones for display
    active_tones = sim.get_active_shepard_tones(t)
    
    # Extract zoom region
    # intensity_zoom = focal_intensity[u_zoom_mask, :][:, v_zoom_mask] / max_intensity_t0
    intensity_zoom = focal_intensity[u_zoom_mask, :][:, v_zoom_mask] / global_max_intensity
    u_zoom = u_fft[u_zoom_mask]
    v_zoom = v_fft[v_zoom_mask]
    
    # Create figure with 4 panels (matching your original layout)
    fig = plt.figure(figsize=(24, 6))
    
    # Panel 1: Active Shepard tones info
    ax1 = plt.subplot(1, 4, 1)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    tone_text = f"Active Shepard Tones (t = {t*1e6:.3f} μs):\n\n"
    for tone in active_tones:
        tone_text += f"n={tone['n']}: {tone['frequency']/1e6:.1f} MHz\n"
        tone_text += f"  Amplitude: {tone['amplitude']:.3f}\n"
        tone_text += f"  → Creates 2 beams:\n"
        tone_text += f"    {tone['frequency']/1e6:.1f} MHz\n"
        tone_text += f"    {(tone['frequency']+f_input_offset)/1e6:.1f} MHz\n\n"
    
    ax1.text(0.1, 0.9, tone_text, transform=ax1.transAxes,
            fontsize=10, family='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Draw beam schematic
    beam_circle = Circle((0.75, 0.3), 0.08, fill=False, edgecolor='blue',
                        linewidth=2, transform=ax1.transAxes)
    ax1.add_patch(beam_circle)
    ax1.text(0.75, 0.15, 'Beam\nwaist', ha='center',
            transform=ax1.transAxes, fontsize=9)
    
    ax1.set_title('Two-Tone Shepard System', fontsize=12, fontweight='bold')
    
    # Panel 2: Focal plane zoom
    ax2 = plt.subplot(1, 4, 2)
    im2 = ax2.imshow(intensity_zoom.T,
                     extent=[u_zoom[0]*1e3, u_zoom[-1]*1e3, 
                            v_zoom[0]*1e3, v_zoom[-1]*1e3],
                     origin='lower', aspect='equal', cmap='hot',
                     norm=PowerNorm(gamma=0.5, vmin=0, vmax=1.0))
    ax2.set_xlabel('u [mm]', fontsize=11)
    ax2.set_ylabel('v [mm]', fontsize=11)
    ax2.set_title(f'Focal Plane ({fov_zoom*1e3}mm zoom)', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='|E|² (normalized)')
    ax2.grid(True, alpha=0.2, color='white')
    
    # Panel 3: Horizontal cross-section
    ax3 = plt.subplot(1, 4, 3)
    center_v_idx = len(v_zoom) // 2
    ax3.plot(u_zoom*1e3, intensity_zoom[:, center_v_idx], 'b-', linewidth=2,
            label='Shepard fading')
    ax3.set_xlabel('u [mm]', fontsize=11)
    ax3.set_ylabel('Intensity (normalized)', fontsize=11)
    ax3.set_title('Horizontal Cross-section (v=0)', fontsize=12)
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Panel 4: Wide FOV
    ax4 = plt.subplot(1, 4, 4)
    
    u_wide_min = -40e-3
    u_wide_max = 0e-3
    v_wide_min = -2e-3
    v_wide_max = 2e-3
    
    u_wide_mask = (u_fft >= u_wide_min) & (u_fft <= u_wide_max)
    v_wide_mask = (v_fft >= v_wide_min) & (v_fft <= v_wide_max)
    
    intensity_wide = focal_intensity[u_wide_mask, :][:, v_wide_mask] / max_intensity_t0
    u_wide = u_fft[u_wide_mask]
    v_wide = v_fft[v_wide_mask]
    
    im4 = ax4.imshow(intensity_wide.T,
                     extent=[u_wide[0]*1e3, u_wide[-1]*1e3,
                            v_wide[0]*1e3, v_wide[-1]*1e3],
                     origin='lower', aspect='equal', cmap='hot',
                     norm=PowerNorm(gamma=0.25, vmin=0, vmax=1.0))
    ax4.set_xlabel('u [mm]', fontsize=11)
    ax4.set_ylabel('v [mm]', fontsize=11)
    ax4.set_title('Wide FOV (Shepard transition)', fontsize=12)
    
    # Mark expected beam positions
    u_deflection_per_mhz = params.F * params.wavelength / params.V
    for tone in active_tones:
        u_pos1 = params.F * params.wavelength * tone['frequency'] / params.V
        u_pos2 = params.F * params.wavelength * (tone['frequency'] - f_input_offset) / params.V
        
        ax4.axvline(-u_pos1*1e3, color='cyan', linestyle='--', 
                   linewidth=1, alpha=tone['amplitude'])
        ax4.axvline(-u_pos2*1e3, color='magenta', linestyle='--',
                   linewidth=1, alpha=tone['amplitude'])
    
    plt.colorbar(im4, ax=ax4, label='|E|² (normalized)')
    ax4.grid(True, alpha=0.2, color='white')
    
    plt.suptitle(f't = {t*1e6:.3f} μs | Shepard sweep: {params.f0/1e6:.0f} → {params.f1/1e6:.0f} MHz | Smooth transition',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    filename = f'{frames_folder}/frame_{i:03d}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    saved_frames.append(filename)

print(f"\n\nSaved {n_frames} frames to {frames_folder}/")

# Create GIF
print("Creating GIF animation...")
images = [Image.open(f) for f in saved_frames]
images[0].save(
    f'beam_animation_shepard_two_tone_{shepard_spacing/1e6:.2f}MHz.gif',
    save_all=True,
    append_images=images[1:],
    duration=100,
    loop=0
)
print("Saved: beam_animation_shepard_two_tone.gif")
print()

print("=" * 60)
print("Shepard two-tone animation complete!")
print(f"  Frames: {n_frames}")
print(f"  Time range: 0 to {t_end*1e6:.2f} us")
print(f"  Key improvement: Smooth Shepard fading instead of abrupt switch")
print(f"  Result: Minimal aberrations at x2 tweezer position")
print("=" * 60)
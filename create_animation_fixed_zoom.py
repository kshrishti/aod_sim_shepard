"""
Create complete animation showing:
- Crystal aperture with frequency distribution
- Focal plane zoom (1.5mm x 1.5mm)
- Horizontal cross-section
All using exact FFT values, centered on t=0 peak
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from aod_simulation import AODParameters, AODSimulation

# Create parameters
params = AODParameters(
    f0=50e6, f1=80e6, alpha=1e12, phi0=0.0,
    V=650.0, wavelength=780e-9, w0=300e-6,
    F=200e-3, M=1.0,
    crystal_length=7.5e-3, beam_position=3.75e-3
)

print("=" * 60)
print("Creating Complete Animation")
print("=" * 60)

# Calculate transit time: time for acoustic wave to cross the beam
transit_time = params.w0 / params.V
print(f"Beam waist w0 = {params.w0*1e6:.1f} um")
print(f"Acoustic velocity V = {params.V:.1f} m/s")
print(f"Transit time T = w0/V = {transit_time*1e6:.3f} us")
print()

sim = AODSimulation(params)

# Create aperture grid
n_points = 400
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

# Step 1: Find peak position at t=0
print("Finding beam peak position at t=0...")
t0 = 0.0
phi_t0 = sim.calculate_phase(XI, ETA, t0, use_frequency_reset=True)
aperture_t0 = gaussian * np.exp(-1j * phi_t0) * quad_phase
field_fft_t0 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(aperture_t0)))
focal_field_t0 = scale * field_fft_t0
focal_intensity_t0 = np.abs(focal_field_t0)**2

u_peak_idx = np.unravel_index(np.argmax(focal_intensity_t0), focal_intensity_t0.shape)[0]
v_peak_idx = np.unravel_index(np.argmax(focal_intensity_t0), focal_intensity_t0.shape)[1]
u_peak_t0 = u_fft[u_peak_idx]
v_peak_t0 = v_fft[v_peak_idx]

print(f"Peak at t=0:")
print(f"  u = {u_peak_t0*1e3:.4f} mm")
print(f"  v = {v_peak_t0*1e3:.4f} mm")

# Get maximum intensity at t=0 for normalization
max_intensity_t0 = np.max(focal_intensity_t0)
print(f"  Max intensity at t=0: {max_intensity_t0:.3e}")
print(f"  (All intensities will be normalized to this = 1.0)")
print()

# Step 2: Define fixed zoom window centered on t=0 peak (1.5mm x 1.5mm)
fov_zoom = 1.5e-3  # 1.5mm
u_zoom_min = u_peak_t0 - fov_zoom/2
u_zoom_max = u_peak_t0 + fov_zoom/2
v_zoom_min = v_peak_t0 - fov_zoom/2
v_zoom_max = v_peak_t0 + fov_zoom/2

u_zoom_mask = (u_fft >= u_zoom_min) & (u_fft <= u_zoom_max)
v_zoom_mask = (v_fft >= v_zoom_min) & (v_fft <= v_zoom_max)

print(f"Fixed zoom window (1.5mm x 1.5mm):")
print(f"  u: [{u_zoom_min*1e3:.4f}, {u_zoom_max*1e3:.4f}] mm")
print(f"  v: [{v_zoom_min*1e3:.4f}, {v_zoom_max*1e3:.4f}] mm")
print(f"  Grid points: {np.sum(u_zoom_mask)} x {np.sum(v_zoom_mask)}")
print()

# Step 3: Create animation frames (from t=0 to t=1.5*T)
# Go beyond T to see the beam fully transition to 50MHz
n_frames = 60
t_end = 1.5 * transit_time
time_array = np.linspace(0, t_end, n_frames)

frames_folder = 'animation_frames_fixed'
os.makedirs(frames_folder, exist_ok=True)
saved_frames = []

print(f"Creating {n_frames} frames (t=0 to t={t_end*1e6:.3f} us = 1.5*T)...")
center_eta = n_points // 2
for i, t in enumerate(time_array):
    print(f"  Frame {i+1}/{n_frames}: t = {t*1e6:.3f} us", end='\r')

    # Calculate phase and frequency at this time
    tau = t - params.T/2 - XI * params.M / params.V
    xi_disc = -params.w0/2 + params.V * t / params.M

    # Calculate instantaneous frequency for visualization
    tau_new = tau
    tau_old = tau + params.chirp_period
    f_new = params.f0 + params.alpha * tau_new
    f_old = params.f0 + params.alpha * tau_old
    frequency = np.where(XI >= xi_disc, f_old, f_new)

    # Calculate phase
    phi = sim.calculate_phase(XI, ETA, t, use_frequency_reset=True)

    # Aperture field
    aperture_field = gaussian * np.exp(-1j * phi)

    # FFT to focal plane
    field_with_prop = aperture_field * quad_phase
    field_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_with_prop)))
    focal_field = scale * field_fft
    focal_intensity = np.abs(focal_field)**2

    # Extract fixed zoom region (exact FFT values) and normalize
    intensity_zoom = focal_intensity[u_zoom_mask, :][:, v_zoom_mask] / max_intensity_t0
    u_zoom = u_fft[u_zoom_mask]
    v_zoom = v_fft[v_zoom_mask]

    # Create figure with 3 panels
    fig = plt.figure(figsize=(18, 6))

    # Panel 1: Frequency in crystal aperture
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(frequency[:, center_eta:center_eta+1].T / 1e6,
                     extent=[xi[0]*1e6, xi[-1]*1e6, -50, 50],
                     origin='lower', aspect='equal', cmap='coolwarm',
                     vmin=50, vmax=80)
    ax1.axvline(xi_disc*1e6, color='red', linestyle='--', linewidth=2, label='Discontinuity')

    # Draw beam waist circle (where the beam is located in the crystal)
    from matplotlib.patches import Circle
    beam_circle = Circle((0, 0), params.w0/2*1e6, fill=False, edgecolor='yellow',
                         linestyle='--', linewidth=2, label='Beam waist')
    ax1.add_patch(beam_circle)

    ax1.set_xlabel('ξ [μm]', fontsize=11)
    ax1.set_ylabel('η [μm]', fontsize=11)
    ax1.set_title('Frequency in Crystal', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(-params.w0/2*1e6*1.5, params.w0/2*1e6*1.5)  # Show region around beam
    cbar1 = plt.colorbar(im1, ax=ax1, label='Frequency [MHz]')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Focal plane zoom (1.5mm) - normalized intensity
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(intensity_zoom.T,
                     extent=[u_zoom[0]*1e3, u_zoom[-1]*1e3, v_zoom[0]*1e3, v_zoom[-1]*1e3],
                     origin='lower', aspect='equal', cmap='hot',
                     norm=plt.matplotlib.colors.PowerNorm(gamma=0.5, vmin=0, vmax=1.0))
    ax2.set_xlabel('u [mm]', fontsize=11)
    ax2.set_ylabel('v [mm]', fontsize=11)
    ax2.set_title('Focal Plane (1.5mm zoom)', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='|E|² (normalized)')
    ax2.grid(True, alpha=0.2, color='white')

    # Panel 3: Horizontal cross-section
    ax3 = plt.subplot(1, 3, 3)
    center_v_idx = len(v_zoom) // 2
    ax3.plot(u_zoom*1e6, intensity_zoom[:, center_v_idx], 'b-', linewidth=2)
    ax3.set_xlabel('u [μm]', fontsize=11)
    ax3.set_ylabel('Intensity |E|² (normalized)', fontsize=11)
    ax3.set_title('Horizontal Cross-section (v=0)', fontsize=12)
    ax3.set_ylim(0, 1.1)  # Fixed scale from 0 to 1
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f't = {t*1e6:.3f} μs | Discontinuity at ξ = {xi_disc*1e6:.0f} μm',
                 fontsize=14, y=0.98)
    plt.tight_layout()

    filename = f'{frames_folder}/frame_{i:03d}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    saved_frames.append(filename)

print(f"\nSaved {n_frames} frames to {frames_folder}/")
print()

# Create GIF
print("Creating GIF animation...")
images = [Image.open(f) for f in saved_frames]
images[0].save(
    'beam_animation_fixed_zoom.gif',
    save_all=True,
    append_images=images[1:],
    duration=100,
    loop=0
)
print("Saved: beam_animation_fixed_zoom.gif")
print()

print("=" * 60)
print("Complete animation finished!")
print(f"  Frames: {n_frames}")
print(f"  Time range: t=0 to t={t_end*1e6:.3f} us (1.5*T)")
print(f"  Transit time T = {transit_time*1e6:.3f} us")
print(f"  Frame location: {frames_folder}/")
print(f"  GIF: beam_animation_fixed_zoom.gif")
print(f"  Zoom: 1.5mm x 1.5mm, centered on u={u_peak_t0*1e3:.4f} mm")
print(f"  Grid points: {np.sum(u_zoom_mask)} x {np.sum(v_zoom_mask)} (exact FFT)")
print("=" * 60)

"""
Visualize beam at a few key times with crystal aperture view
"""
import numpy as np
import matplotlib.pyplot as plt
from aod_simulation import AODParameters, AODSimulation

# Create parameters
params = AODParameters(
    f0=50e6, f1=80e6, alpha=1e12, phi0=0.0,
    V=650.0, wavelength=780e-9, w0=300e-6,
    F=200e-3, M=1.0,
    crystal_length=7.5e-3, beam_position=3.75e-3
)

print("=" * 60)
print("Testing a Few Key Times")
print("=" * 60)
print(f"Transit time T = {params.T*1e6:.3f} us")
print()

sim = AODSimulation(params)

# Key time points
times = [0.0, params.T/4, params.T/2, 3*params.T/4, params.T]
time_labels = ['t=0 (left edge)', 't=T/4', 't=T/2 (center)', 't=3T/4', 't=T (right edge)']

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

# Process each time
for idx, (t, label) in enumerate(zip(times, time_labels)):
    print(f"Computing {label}...")

    # Calculate phase and frequency at this time
    tau = t - params.T/2 - XI * params.M / params.V
    xi_disc = -params.w0/2 + params.V * t / params.M

    # Calculate instantaneous frequency
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

    # Find beam center in focal plane
    u_peak_idx = np.unravel_index(np.argmax(focal_intensity), focal_intensity.shape)[0]
    u_peak = u_fft[u_peak_idx]

    # Extract 1mm zoom around peak
    fov_zoom = 1.0e-3
    u_zoom_mask = (u_fft >= u_peak - fov_zoom/2) & (u_fft <= u_peak + fov_zoom/2)
    v_zoom_mask = (v_fft >= -fov_zoom/2) & (v_fft <= fov_zoom/2)

    intensity_zoom = focal_intensity[u_zoom_mask, :][:, v_zoom_mask]
    u_zoom = u_fft[u_zoom_mask]
    v_zoom = v_fft[v_zoom_mask]

    # Create figure with 3 panels
    fig = plt.figure(figsize=(18, 6))

    # Panel 1: Frequency in crystal aperture
    ax1 = plt.subplot(1, 3, 1)
    center_eta = n_points // 2
    im1 = ax1.imshow(frequency[:, center_eta:center_eta+1].T / 1e6,
                     extent=[xi[0]*1e6, xi[-1]*1e6, -50, 50],
                     origin='lower', aspect='auto', cmap='coolwarm',
                     vmin=50, vmax=80)
    ax1.axvline(xi_disc*1e6, color='red', linestyle='--', linewidth=2, label='Discontinuity')
    ax1.axvline(-params.w0/2*1e6, color='white', linestyle=':', alpha=0.5)
    ax1.axvline(params.w0/2*1e6, color='white', linestyle=':', alpha=0.5)
    ax1.set_xlabel('xi [um]')
    ax1.set_ylabel('eta [um]')
    ax1.set_title(f'Frequency in Crystal | {label}')
    ax1.legend()
    cbar1 = plt.colorbar(im1, ax=ax1, label='Frequency [MHz]')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Focal plane zoom (1mm)
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(intensity_zoom.T,
                     extent=[u_zoom[0]*1e3, u_zoom[-1]*1e3, v_zoom[0]*1e3, v_zoom[-1]*1e3],
                     origin='lower', aspect='equal', cmap='hot',
                     norm=plt.matplotlib.colors.PowerNorm(gamma=0.5))
    ax2.set_xlabel('u [mm]')
    ax2.set_ylabel('v [mm]')
    ax2.set_title(f'Focal Plane (1mm zoom)')
    plt.colorbar(im2, ax=ax2, label='|E|^2')
    ax2.grid(True, alpha=0.2, color='white')

    # Panel 3: Horizontal cross-section
    ax3 = plt.subplot(1, 3, 3)
    center_v_idx = len(v_zoom) // 2
    ax3.plot(u_zoom*1e6, intensity_zoom[:, center_v_idx], 'b-', linewidth=2)
    ax3.set_xlabel('u [um]')
    ax3.set_ylabel('Intensity |E|^2')
    ax3.set_title(f'Horizontal Cross-section (v=0)')
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f't = {t*1e6:.3f} us | Discontinuity at xi = {xi_disc*1e6:.0f} um',
                 fontsize=14, y=1.00)
    plt.tight_layout()

    filename = f'test_time_{idx}.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

print()
print("=" * 60)
print("Done! Check the images:")
for idx, label in enumerate(time_labels):
    print(f"  test_time_{idx}.png - {label}")
print("=" * 60)

"""
Visualize beam profile in both AOD crystal and focal plane
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
print("Beam Profile Visualization")
print("=" * 60)
print(f"Beam waist in crystal: {params.w0*1e6:.0f} um")
print(f"Diffraction-limited spot: {params.wavelength*params.F/params.w0*1e6:.0f} um")
print(f"Transit time: {params.T*1e6:.3f} us")
print()

# Time to simulate (set this value)
t_sim = 0.0  # 0.0 for left edge, T/2 for center, T for right edge
xi_disc = -params.w0/2 + params.V * t_sim / params.M

print(f"Simulating at t = {t_sim*1e6:.3f} us (t/T = {t_sim/params.T:.2f})")
print(f"Discontinuity position: xi = {xi_disc*1e6:.1f} um")
print()

# ===== PART 1: Beam in AOD crystal =====
print("Computing field in AOD crystal (aperture)...")

n_points = 400
xi_max = 3 * params.w0  # Show +/- 3 beam waists
xi = np.linspace(-xi_max, xi_max, n_points)
eta = np.linspace(-xi_max, xi_max, n_points)
XI, ETA = np.meshgrid(xi, eta, indexing='ij')

# Create simulation object to use frequency reset model
sim = AODSimulation(params)

# Calculate phase using frequency reset model
phi = sim.calculate_phase(XI, np.zeros_like(XI), t_sim, use_frequency_reset=True)

# Gaussian envelope
gaussian = np.exp(-(XI**2 + ETA**2) * params.M**2 / params.w0**2)

# Calculate field in aperture
aperture_field = gaussian * np.exp(-1j * phi)

print("Done!")
print()

# ===== PART 2: Beam in focal plane =====
print("Computing field in focal plane (after Fourier transform)...")

# FFT to get focal plane field
quad_phase = np.exp(-1j * params.k * (XI**2 + ETA**2) / (2 * params.F))
field_with_prop = aperture_field * quad_phase

# Take FFT
field_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_with_prop)))

# Create frequency arrays
d_xi = xi[1] - xi[0]
freq_xi = np.fft.fftshift(np.fft.fftfreq(n_points, d_xi))
freq_eta = np.fft.fftshift(np.fft.fftfreq(n_points, d_xi))

# Convert to focal plane coordinates
u_fft = params.wavelength * params.F * freq_xi
v_fft = params.wavelength * params.F * freq_eta

# Scale field
scale = d_xi * d_xi
focal_field = scale * field_fft

focal_intensity = np.abs(focal_field)**2

print("Done!")
print()

# ===== PLOTTING =====
fig = plt.figure(figsize=(16, 12))

# ===== ROW 1: AOD Crystal =====
# Aperture intensity
ax1 = plt.subplot(3, 3, 1)
im = ax1.imshow(np.abs(aperture_field).T**2,
                extent=[xi[0]*1e6, xi[-1]*1e6, eta[0]*1e6, eta[-1]*1e6],
                origin='lower', aspect='equal', cmap='hot')
ax1.set_xlabel('xi [um]')
ax1.set_ylabel('eta [um]')
ax1.set_title('Intensity in AOD crystal')
plt.colorbar(im, ax=ax1, label='|E|^2')
ax1.grid(True, alpha=0.2, color='white')

# Aperture phase
ax2 = plt.subplot(3, 3, 2)
im = ax2.imshow(np.angle(aperture_field).T,
                extent=[xi[0]*1e6, xi[-1]*1e6, eta[0]*1e6, eta[-1]*1e6],
                origin='lower', aspect='equal', cmap='twilight',
                vmin=-np.pi, vmax=np.pi)
ax2.set_xlabel('xi [um]')
ax2.set_ylabel('eta [um]')
ax2.set_title('Phase in AOD crystal')
plt.colorbar(im, ax=ax2, label='arg(E) [rad]')
ax2.grid(True, alpha=0.2, color='white')

# Phase profile along xi
ax3 = plt.subplot(3, 3, 3)
center = n_points // 2
ax3.plot(xi*1e6, phi[:, center]/(2*np.pi), 'b-', linewidth=2)
ax3.set_xlabel('xi [um]')
ax3.set_ylabel('Phase [cycles]')
ax3.set_title('Phase along xi axis (eta=0)')
ax3.grid(True, alpha=0.3)

# ===== ROW 2: Focal Plane (full FOV) =====
# Full focal plane
ax4 = plt.subplot(3, 3, 4)
im = ax4.imshow(focal_intensity.T,
                extent=[u_fft[0]*1e3, u_fft[-1]*1e3, v_fft[0]*1e3, v_fft[-1]*1e3],
                origin='lower', aspect='equal', cmap='hot',
                norm=plt.matplotlib.colors.PowerNorm(gamma=0.5))
ax4.set_xlabel('u [mm]')
ax4.set_ylabel('v [mm]')
ax4.set_title('Focal plane intensity (full FFT range)')
plt.colorbar(im, ax=ax4, label='|E|^2')
ax4.grid(True, alpha=0.2, color='white')

# Find beam center
u_center_idx = np.unravel_index(np.argmax(focal_intensity), focal_intensity.shape)[0]
v_center_idx = np.unravel_index(np.argmax(focal_intensity), focal_intensity.shape)[1]
u_center = u_fft[u_center_idx]
v_center = v_fft[v_center_idx]
print(f"Beam center in focal plane: u={u_center*1e3:.3f} mm, v={v_center*1e3:.3f} mm")

# ===== ROW 3: Focal Plane (zoomed to 1mm x 1mm) =====
# Find indices for 1mm FOV around beam center
fov_size = 1e-3  # 1 mm
u_mask = (u_fft >= u_center - fov_size/2) & (u_fft <= u_center + fov_size/2)
v_mask = (v_fft >= v_center - fov_size/2) & (v_fft <= v_center + fov_size/2)

u_zoom = u_fft[u_mask]
v_zoom = v_fft[v_mask]
intensity_zoom = focal_intensity[u_mask, :][:, v_mask]

ax5 = plt.subplot(3, 3, 7)
im = ax5.imshow(intensity_zoom.T,
                extent=[u_zoom[0]*1e3, u_zoom[-1]*1e3, v_zoom[0]*1e3, v_zoom[-1]*1e3],
                origin='lower', aspect='equal', cmap='hot',
                norm=plt.matplotlib.colors.PowerNorm(gamma=0.5))
ax5.set_xlabel('u [mm]')
ax5.set_ylabel('v [mm]')
ax5.set_title('Focal plane intensity (1mm x 1mm FOV)')
plt.colorbar(im, ax=ax5, label='|E|^2')
ax5.grid(True, alpha=0.2, color='white')

# Cross-section horizontal
ax6 = plt.subplot(3, 3, 8)
v_center_zoom_idx = len(v_zoom) // 2
ax6.plot(u_zoom*1e6, intensity_zoom[:, v_center_zoom_idx], 'b-', linewidth=2)
ax6.set_xlabel('u [um]')
ax6.set_ylabel('Intensity')
ax6.set_title('Horizontal cross-section (v=0)')
ax6.grid(True, alpha=0.3)

# Cross-section vertical
ax7 = plt.subplot(3, 3, 9)
u_center_zoom_idx = len(u_zoom) // 2
ax7.plot(v_zoom*1e6, intensity_zoom[u_center_zoom_idx, :], 'r-', linewidth=2)
ax7.set_xlabel('v [um]')
ax7.set_ylabel('Intensity')
ax7.set_title('Vertical cross-section (u=0)')
ax7.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('beam_profile_complete.png', dpi=200, bbox_inches='tight')
print()
print("Saved: beam_profile_complete.png")
print("=" * 60)

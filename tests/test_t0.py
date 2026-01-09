"""
Quick test to visualize beam profile at t=0
"""
import numpy as np
import matplotlib.pyplot as plt
from aod_simulation import AODParameters, AODSimulation

# Create parameters
params = AODParameters(
    f0=50e6,           # 50 MHz
    f1=80e6,           # 80 MHz
    alpha=1e12,        # 1 MHz/μs chirp rate
    phi0=0.0,
    V=650.0,
    wavelength=780e-9,
    w0=300e-6,
    F=200e-3,
    M=1.0,
    crystal_length=7.5e-3,
    beam_position=3.75e-3
)

print("=" * 60)
print("AOD Beam Profile at t=0")
print("=" * 60)
print(f"Crystal length: {params.crystal_length*1e3:.1f} mm")
print(f"Beam waist: {params.w0*1e6:.0f} um")
print(f"Transit time: {params.T*1e6:.3f} us")
print(f"Frequency range: {params.f0/1e6:.1f} - {params.f1/1e6:.1f} MHz")
print(f"Chirp rate: {params.alpha/1e12:.3f} MHz/us")
print(f"Phase offset for continuity: {params.phi0_for_continuity/(2*np.pi):.2f} cycles")
print()

# Create simulation
sim = AODSimulation(params)

# Define focal plane region
u_max = params.F * params.wavelength * params.f1 / params.V
spot_size = params.wavelength * params.F / params.w0

u_center = -u_max
v_center = 0
fov_size = 10 * spot_size

u_range = (u_center - fov_size/2, u_center + fov_size/2)
v_range = (v_center - fov_size/2, v_center + fov_size/2)

print(f"Focal plane FOV: {fov_size*1e3:.3f} mm ({fov_size/spot_size:.0f} spot sizes)")
print(f"Diffraction-limited spot: {spot_size*1e6:.1f} um")
print()

# Simulate at t=0 (reset just entering beam)
print("Computing field at t=0 (reset at left edge of beam)...")
u = np.linspace(u_range[0], u_range[1], 300)
v = np.linspace(v_range[0], v_range[1], 300)

field_t0 = sim.compute_field_grid(u, v, t=0.0, n_points=400)
intensity_t0 = np.abs(field_t0)**2
phase_t0 = np.angle(field_t0)

print("Done!")
print()

# Also compute at t = -T (before reset reaches beam)
print("Computing field at t=-T (before reset reaches beam)...")
field_pre = sim.compute_field_grid(u, v, t=-params.T, n_points=400)
intensity_pre = np.abs(field_pre)**2
phase_pre = np.angle(field_pre)
print("Done!")
print()

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# t = -T: Before reset
ax = axes[0, 0]
im = ax.imshow(intensity_pre.T, extent=[u[0]*1e3, u[-1]*1e3, v[0]*1e3, v[-1]*1e3],
               origin='lower', aspect='auto', cmap='hot',
               norm=plt.matplotlib.colors.PowerNorm(gamma=0.5))
ax.set_xlabel('u [mm]')
ax.set_ylabel('v [mm]')
ax.set_title(f'Intensity at t = -T (before reset reaches beam)')
plt.colorbar(im, ax=ax, label='|E|²')

ax = axes[0, 1]
im = ax.imshow(phase_pre.T, extent=[u[0]*1e3, u[-1]*1e3, v[0]*1e3, v[-1]*1e3],
               origin='lower', aspect='auto', cmap='twilight',
               vmin=-np.pi, vmax=np.pi)
ax.set_xlabel('u [mm]')
ax.set_ylabel('v [mm]')
ax.set_title(f'Phase at t = -T')
plt.colorbar(im, ax=ax, label='arg(E) [rad]')

# t = 0: Reset at left edge
ax = axes[1, 0]
im = ax.imshow(intensity_t0.T, extent=[u[0]*1e3, u[-1]*1e3, v[0]*1e3, v[-1]*1e3],
               origin='lower', aspect='auto', cmap='hot',
               norm=plt.matplotlib.colors.PowerNorm(gamma=0.5))
ax.set_xlabel('u [mm]')
ax.set_ylabel('v [mm]')
ax.set_title(f'Intensity at t = 0 (reset at left edge of beam)')
plt.colorbar(im, ax=ax, label='|E|²')

ax = axes[1, 1]
im = ax.imshow(phase_t0.T, extent=[u[0]*1e3, u[-1]*1e3, v[0]*1e3, v[-1]*1e3],
               origin='lower', aspect='auto', cmap='twilight',
               vmin=-np.pi, vmax=np.pi)
ax.set_xlabel('u [mm]')
ax.set_ylabel('v [mm]')
ax.set_title(f'Phase at t = 0')
plt.colorbar(im, ax=ax, label='arg(E) [rad]')

plt.tight_layout()
plt.savefig('beam_profile_t0_comparison.png', dpi=200, bbox_inches='tight')
print("Saved: beam_profile_t0_comparison.png")

# Cross-sections
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

center_v = len(v) // 2

ax = axes[0]
ax.plot(u*1e3, intensity_pre[:, center_v], 'b-', linewidth=2, label='t = -T (before reset)')
ax.plot(u*1e3, intensity_t0[:, center_v], 'r-', linewidth=2, label='t = 0 (reset at left edge)')
ax.set_xlabel('u [mm]')
ax.set_ylabel('Intensity |E|²')
ax.set_title('Horizontal cross-section (v=0)')
ax.legend()
ax.grid(True, alpha=0.3)

center_u = len(u) // 2

ax = axes[1]
ax.plot(v*1e3, intensity_pre[center_u, :], 'b-', linewidth=2, label='t = -T (before reset)')
ax.plot(v*1e3, intensity_t0[center_u, :], 'r-', linewidth=2, label='t = 0 (reset at left edge)')
ax.set_xlabel('v [mm]')
ax.set_ylabel('Intensity |E|²')
ax.set_title('Vertical cross-section (u=0)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('beam_profile_t0_crosssections.png', dpi=200, bbox_inches='tight')
print("Saved: beam_profile_t0_crosssections.png")

print()
print("=" * 60)
print("At t=0:")
print(f"  - Reset discontinuity is at xi = {-params.w0/2*1e6:.1f} um (left edge)")
print(f"  - Fraction of beam in OLD region: ~100%")
print(f"  - Fraction of beam in NEW region: ~0%")
print(f"  → Beam profile should be nearly identical to steady-state chirp")
print("=" * 60)

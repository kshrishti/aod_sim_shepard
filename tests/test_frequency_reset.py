"""
Test the frequency reset implementation
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
print("Frequency Reset Test")
print("=" * 60)
print(f"Chirp range: {params.f0/1e6:.0f} - {params.f1/1e6:.0f} MHz")
print(f"Chirp rate: {params.alpha/1e12:.3f} MHz/us")
print(f"Chirp period: {params.chirp_period*1e6:.1f} us")
print(f"Beam transit time: {params.T*1e6:.3f} us")
print()

sim = AODSimulation(params)

# Create 1D array along xi axis (eta=0)
n_points = 1000
xi = np.linspace(-1.5*params.w0, 1.5*params.w0, n_points)
eta = np.zeros_like(xi)

# Test at t=0 (reset just entering at left edge)
t = 0.0

# Calculate phase
phi = sim.calculate_phase(xi, eta, t, use_frequency_reset=True)

# Calculate instantaneous frequency from phase gradient
# f = (1/2π) × dφ/dt, but we have dφ/dxi, so we need chain rule
# dφ/dt = (dφ/dxi) × (dxi/dt) where dxi/dt relates to tau
# Actually easier: calculate tau and frequency directly
tau = t - params.T/2 - xi * params.M / params.V
xi_disc = -params.w0/2 + params.V * t / params.M

# Calculate frequency for each region
tau_new = tau
tau_old = tau + params.chirp_period

f_new = params.f0 + params.alpha * tau_new
f_old = params.f0 + params.alpha * tau_old

# Select frequency based on region
frequency = np.where(xi >= xi_disc, f_old, f_new)

# Find the frequency range in each region
new_region_mask = xi < xi_disc
old_region_mask = xi >= xi_disc

if np.any(new_region_mask):
    f_new_min, f_new_max = frequency[new_region_mask].min(), frequency[new_region_mask].max()
    print("NEW region (post-reset):")
    print(f"  Frequency range: {f_new_min/1e6:.2f} - {f_new_max/1e6:.2f} MHz")
    print(f"  Width: {(f_new_max - f_new_min)/1e6:.3f} MHz")
    print()

if np.any(old_region_mask):
    f_old_min, f_old_max = frequency[old_region_mask].min(), frequency[old_region_mask].max()
    print("OLD region (pre-reset):")
    print(f"  Frequency range: {f_old_min/1e6:.2f} - {f_old_max/1e6:.2f} MHz")
    print(f"  Width: {(f_old_max - f_old_min)/1e6:.3f} MHz")
    print()

print(f"Discontinuity at xi = {xi_disc*1e6:.1f} um")
print()

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Phase profile
ax = axes[0]
ax.plot(xi*1e6, phi/(2*np.pi), 'b-', linewidth=2)
ax.axvline(xi_disc*1e6, color='r', linestyle='--', linewidth=2, label='Discontinuity')
ax.axvline(-params.w0/2*1e6, color='gray', linestyle=':', alpha=0.5, label='Beam edges')
ax.axvline(params.w0/2*1e6, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('xi [um]')
ax.set_ylabel('Phase [cycles]')
ax.set_title(f'Phase profile at t=0 (reset at left edge)')
ax.grid(True, alpha=0.3)
ax.legend()

# Frequency profile
ax = axes[1]
ax.plot(xi*1e6, frequency/1e6, 'g-', linewidth=2)
ax.axvline(xi_disc*1e6, color='r', linestyle='--', linewidth=2, label='Discontinuity')
ax.axvline(-params.w0/2*1e6, color='gray', linestyle=':', alpha=0.5, label='Beam edges')
ax.axvline(params.w0/2*1e6, color='gray', linestyle=':', alpha=0.5)
ax.axhline(params.f0/1e6, color='blue', linestyle=':', alpha=0.5, label=f'f0 = {params.f0/1e6:.0f} MHz')
ax.axhline(params.f1/1e6, color='red', linestyle=':', alpha=0.5, label=f'f1 = {params.f1/1e6:.0f} MHz')
ax.set_xlabel('xi [um]')
ax.set_ylabel('Frequency [MHz]')
ax.set_title('Instantaneous frequency across beam')
ax.set_ylim([45, 85])
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('frequency_reset_test.png', dpi=200, bbox_inches='tight')
print("Saved: frequency_reset_test.png")
print("=" * 60)

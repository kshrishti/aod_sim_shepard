"""
Parameter Exploration Script for AOD Simulation

This script makes it easy to study how different parameters affect the
beam profile during phase discontinuity propagation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from aod_simulation import AODParameters, AODSimulation, plot_time_snapshot
from aod_simulation_shepard import AODParameters, AODSimulation, plot_time_snapshot
import os


def explore_chirp_rate(chirp_rates_MHz_per_us, n_time_steps=15):
    """
    Study the effect of different chirp rates

    Args:
        chirp_rates_MHz_per_us: List of chirp rates in MHz/μs
        n_time_steps: Number of time steps to simulate
    """
    print("=" * 70)
    print("EXPLORING CHIRP RATE EFFECTS")
    print("=" * 70)
    
    base_params = AODParameters(
        f0=50e6,
        f1=80e6,
        alpha=1e12,  # Will be overridden
        wavelength=780e-9,
        w0=300e-6,
        F=200e-3,
        V=650.0,
        M=1.0
    )
    
    results_list = []
    
    for i, alpha_MHz_us in enumerate(chirp_rates_MHz_per_us):
        print(f"\n--- Chirp rate {i+1}/{len(chirp_rates_MHz_per_us)}: {alpha_MHz_us:.3f} MHz/μs ---")
        
        # Update chirp rate
        params = AODParameters(
            f0=base_params.f0,
            f1=base_params.f1,
            alpha=alpha_MHz_us * 1e12,  # Convert to Hz/s
            wavelength=base_params.wavelength,
            w0=base_params.w0,
            F=base_params.F,
            V=base_params.V,
            M=base_params.M
        )
        
        print(f"  Chirp period: {params.chirp_period*1e6:.2f} μs")
        print(f"  Transit time: {params.T*1e6:.3f} μs")
        print(f"  Frequency change during transit: {params.alpha * params.T / 1e6:.3f} MHz")
        
        # Run simulation
        sim = AODSimulation(params)
        
        u_max = params.F * params.wavelength * params.f1 / params.V
        spot_size = params.wavelength * params.F / params.w0
        
        # Zoom in: show ±5 spot sizes around the deflected beam
        u_center = -u_max
        v_center = 0
        fov_size = 10 * spot_size
        
        u_range = (u_center - fov_size/2, u_center + fov_size/2)
        v_range = (v_center - fov_size/2, v_center + fov_size/2)
        
        results = sim.simulate_time_series(
            n_time_steps=n_time_steps,
            u_range=u_range,
            v_range=v_range,
            n_spatial_points=300,
            n_integration_points=400
        )
        
        results_list.append(results)
        
        # Plot middle time point
        mid_idx = n_time_steps // 2
        fig = plot_time_snapshot(results, mid_idx, show_phase=True)
        fig.suptitle(f'Chirp rate: {alpha_MHz_us:.3f} MHz/μs', fontsize=14, y=1.02)
        # plt.savefig(f'chirp_comparison_{i}.png', dpi=200, bbox_inches='tight')
        plt.savefig(f'chirp_comparison_shepard_{i}.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    # Create comparison plot
    fig, axes = plt.subplots(len(chirp_rates_MHz_per_us), 1, 
                            figsize=(10, 4*len(chirp_rates_MHz_per_us)))
    
    if len(chirp_rates_MHz_per_us) == 1:
        axes = [axes]
    
    for i, (results, alpha) in enumerate(zip(results_list, chirp_rates_MHz_per_us)):
        mid_idx = n_time_steps // 2
        u = results['u'] * 1e3
        intensity = results['intensity'][mid_idx]
        
        # Plot horizontal cross-section
        center_v = intensity.shape[1] // 2
        axes[i].plot(u, intensity[:, center_v], 'b-', linewidth=2)
        axes[i].set_ylabel('Intensity')
        axes[i].set_title(f'α = {alpha:.3f} MHz/μs (mid-transit)')
        axes[i].grid(True, alpha=0.3)
        
        if i == len(chirp_rates_MHz_per_us) - 1:
            axes[i].set_xlabel('u [mm]')
    
    plt.tight_layout()
    # plt.savefig('chirp_rate_comparison.png', dpi=200, bbox_inches='tight')
    plt.savefig('chirp_rate_shepard_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 70)
    print("Chirp rate exploration complete!")
    print("=" * 70)
    
    return results_list


# Note: explore_phase_discontinuity function removed - phase offset is now
# automatically calculated for continuity at frequency reset


def create_animation(results, output_path='aod_animation.mp4', fps=10):
    """
    Create an animation showing the temporal evolution
    
    Args:
        results: Dictionary from simulate_time_series
        output_path: Path to save animation
        fps: Frames per second
    """
    print("\nCreating animation...")
    
    u = results['u'] * 1e3  # Convert to mm
    v = results['v'] * 1e3
    intensity = results['intensity']
    phase = results['phase']
    time = results['time'] * 1e6  # Convert to μs
    T = results['params'].T * 1e6
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Initialize plots
    im1 = ax1.imshow(intensity[0].T, extent=[u[0], u[-1], v[0], v[-1]],
                     origin='lower', aspect='auto', cmap='hot',
                     norm=plt.matplotlib.colors.PowerNorm(gamma=0.5))
    ax1.set_xlabel('u [mm]')
    ax1.set_ylabel('v [mm]')
    title1 = ax1.set_title('')
    plt.colorbar(im1, ax=ax1, label='|E|²')
    
    im2 = ax2.imshow(phase[0].T, extent=[u[0], u[-1], v[0], v[-1]],
                     origin='lower', aspect='auto', cmap='twilight',
                     vmin=-np.pi, vmax=np.pi)
    ax2.set_xlabel('u [mm]')
    ax2.set_ylabel('v [mm]')
    title2 = ax2.set_title('')
    plt.colorbar(im2, ax=ax2, label='arg(E) [rad]')
    
    plt.tight_layout()
    
    def update(frame):
        im1.set_array(intensity[frame].T)
        im2.set_array(phase[frame].T)
        title1.set_text(f'Intensity at t = {time[frame]:.3f} μs (t/T = {time[frame]/T:.2f})')
        title2.set_text(f'Phase at t = {time[frame]:.3f} μs')
        return [im1, im2, title1, title2]
    
    anim = FuncAnimation(fig, update, frames=len(time), blit=True)
    
    # Save animation
    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=fps)
    anim.save(output_path, writer=writer)
    
    plt.close()
    
    print(f"Animation saved to {output_path}")


def quick_simulation(chirp_rate_MHz_us=1.0,
                     n_time_steps=20, make_animation=False):
    """
    Quick simulation with specified parameters

    Args:
        chirp_rate_MHz_us: Chirp rate [MHz/μs]
        n_time_steps: Number of time steps
        make_animation: Whether to create animation
    """
    print("=" * 70)
    print("QUICK SIMULATION")
    print("=" * 70)
    print(f"Chirp rate: {chirp_rate_MHz_us:.3f} MHz/μs")
    print("=" * 70)
    
    params = AODParameters(
        f0=50e6,
        f1=80e6,
        alpha=chirp_rate_MHz_us * 1e12,
        wavelength=780e-9,
        w0=300e-6,
        F=200e-3,
        V=650.0,
        M=1.0
    )
    
    sim = AODSimulation(params)
    
    u_max = params.F * params.wavelength * params.f1 / params.V
    spot_size = params.wavelength * params.F / params.w0
    
    # Zoom in: show ±5 spot sizes around the deflected beam
    u_center = -u_max
    v_center = 0
    fov_size = 10 * spot_size
    
    u_range = (u_center - fov_size/2, u_center + fov_size/2)
    v_range = (v_center - fov_size/2, v_center + fov_size/2)

    results = sim.simulate_time_series(
        n_time_steps=n_time_steps,
        u_range=u_range,
        v_range=v_range,
        n_spatial_points=300,
        n_integration_points=400
    )
    
    # Save snapshots at key times
    for i, frac in enumerate([0, 0.25, 0.5, 0.75, 1.0]):
        idx = int(frac * (n_time_steps - 1))
        fig = plot_time_snapshot(results, idx, show_phase=True)
        plt.savefig(f'quick_sim_t{i}.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    if make_animation:
        create_animation(results)
    
    print("\n" + "=" * 70)
    print("Quick simulation complete!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "chirp":
            # Explore different chirp rates
            chirp_rates = [0.5, 1.0, 2.0, 5.0]  # MHz/μs
            explore_chirp_rate(chirp_rates)

        elif mode == "quick":
            # Quick simulation with default or specified parameters
            chirp = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
            quick_simulation(chirp, n_time_steps=20, make_animation=False)
            
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python explore_parameters.py [chirp|quick] [chirp_rate_MHz_us]")
    else:
        # Default: run quick simulation
        print("Running default quick simulation...")
        print("Usage: python explore_parameters.py [chirp|quick] [chirp_rate_MHz_us]")
        print()
        quick_simulation(chirp_rate_MHz_us=1.0, n_time_steps=15)

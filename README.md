# AOD Phase Discontinuity Simulation

## Overview

This package simulates the electric field in the focal plane of a lens after an Acousto-Optic Deflector (AOD) when a **phase discontinuity** propagates through the beam. This occurs when the RF drive signal is phase-reset while continuously chirping.

## Physical Setup

```
[Laser Beam] → [AOD with chirped RF] → [Lens] → [Focal Plane]
                     ↓
              Phase discontinuity 
              propagates at V = 650 m/s
```

### Key Physics

1. **Continuous chirp**: RF frequency sweeps from f₀ to f₁ continuously
2. **Phase reset**: At t=0, the phase jumps by Δφ while maintaining chirp rate α
3. **Propagating discontinuity**: The phase step travels through the crystal at acoustic velocity V
4. **Time-dependent beam profile**: As the discontinuity crosses the beam, it creates interference patterns in the focal plane

### Mathematical Model

The electric field in the focal plane is:

```
E_out(u,v,t) ∝ ∫∫ exp(-jφ(ξ,η,t)) × exp(-(ξ²+η²)M²/w₀²) × 
                  exp(-jk(ξ²+η²)/(2F)) × exp(-j2π(uξ+vη)/(λF)) dξdη
```

where the phase includes the propagating discontinuity:

```
φ(ξ,η,t) = 2π[f₀·τ + (α/2)·τ²] + φ₀     for ξ < ξ_disc(t)
φ(ξ,η,t) = 2π[f₀·τ + (α/2)·τ²] + φ₀ - Δφ for ξ ≥ ξ_disc(t)

where:
  τ = t - T/2 - ξM/V  (acoustic propagation delay)
  ξ_disc(t) = -w₀/2 + Vt/M  (discontinuity position)
```

## Default Parameters

```python
f₀ = 50 MHz          # Start frequency
f₁ = 80 MHz          # End frequency
α = 1 MHz/μs         # Chirp rate (adjustable)
V = 650 m/s          # Acoustic velocity (TeO₂)
w₀ = 300 μm          # Beam waist
λ = 780 nm           # Wavelength
F = 200 mm           # Focal length
M = 1.0              # Magnification
T = w₀/V ≈ 0.46 μs   # Transit time
Δφ = 2π              # Phase discontinuity (1 cycle)
```

## Repository Structure

```
aod_sim_chirp/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore patterns
├── aod_simulation.py                 # Core simulation engine
├── create_animation_fixed_zoom.py    # Single-tone animation script
├── create_animation_two_tone.py      # Two-tone animation script
├── test_few_times.py                 # Quick test at multiple time points
├── test_t0.py                        # t=0 vs t=-T comparison
└── explore_parameters.py             # Parameter sweep utilities
```

## Installation

Required packages:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy scipy matplotlib Pillow
```

## Usage

### 1. Basic Simulation

Run the default simulation:
```bash
python aod_simulation.py
```

This generates:
- 5 time snapshots showing intensity and phase evolution (high resolution: 300×300 pixels)
- Field of view: 10 diffraction-limited spot sizes centered on the beam
- Cross-section plots through beam center
- Analysis of beam properties (centroid, width, peak intensity)

**Resolution settings:**
- Focal plane grid: 300×300 points (up from default 150×150)
- Integration grid: 400×400 points (up from default 200×200)
- Field of view: Automatically sized to 10× the diffraction-limited spot size

### 2. Quick Custom Simulation

```bash
python explore_parameters.py quick [chirp_rate] [phase_discontinuity]
```

Examples:
```bash
python explore_parameters.py quick 1.0 1.0      # α=1 MHz/μs, Δφ=2π
python explore_parameters.py quick 2.0 0.5      # α=2 MHz/μs, Δφ=π
python explore_parameters.py quick 5.0 2.0      # α=5 MHz/μs, Δφ=4π
```

### 3. Parameter Exploration

**Explore chirp rate effects:**
```bash
python explore_parameters.py chirp
```
This tests α = [0.5, 1.0, 2.0, 5.0] MHz/μs

**Explore phase discontinuity effects:**
```bash
python explore_parameters.py phase
```
This tests Δφ = [0.25, 0.5, 1.0, 2.0] × 2π

### 4. Python API

```python
from aod_simulation import AODParameters, AODSimulation

# Create custom parameters
params = AODParameters(
    f0=50e6,
    f1=80e6,
    alpha=2e12,        # 2 MHz/μs
    wavelength=780e-9,
    w0=300e-6,
    F=200e-3,
    V=650.0,
    M=1.0
)

# Run simulation
sim = AODSimulation(params)

u_max = params.F * params.wavelength * params.f1 / params.V
results = sim.simulate_time_series(
    n_time_steps=20,
    delta_phi=2*np.pi,
    u_range=(-2*u_max, 2*u_max),
    v_range=(-2*u_max, 2*u_max),
    n_spatial_points=150,
    n_integration_points=200
)

# Access results
time = results['time']          # Time array [s]
u = results['u']                # u coordinates [m]
v = results['v']                # v coordinates [m]
field = results['field']        # Complex field [time, u, v]
intensity = results['intensity'] # Intensity [time, u, v]
phase = results['phase']        # Phase [time, u, v]

# Analyze
from aod_simulation import analyze_beam_properties
props = analyze_beam_properties(results, time_index=10)
print(f"Centroid: ({props['u_centroid']*1e3:.2f}, {props['v_centroid']*1e3:.2f}) mm")
print(f"Beam width: {props['u_width']*1e6:.1f} μm × {props['v_width']*1e6:.1f} μm")
```

## Key Features

### 1. Full 2D Simulation
- Computes both intensity and phase in focal plane
- Accounts for Gaussian beam profile
- Includes quadratic phase from propagation

### 2. Accurate Phase Model
- Time-delayed phase due to acoustic propagation
- Discontinuity position tracks through beam
- Continuous chirp maintained throughout

### 3. Efficient Computation
- FFT-based integration for speed
- Vectorized calculations
- Adjustable resolution vs accuracy trade-off

### 4. Comprehensive Analysis
- Beam centroid tracking
- RMS width calculation
- Peak intensity evolution
- Cross-section extraction

## Physical Insights

### What Happens During Transit?

1. **t = 0** (Entry): Discontinuity enters beam at left edge
   - Beam starts to develop asymmetry
   - Small interference fringes appear

2. **t = T/4**: Discontinuity 1/4 through beam
   - Clear separation into two phase regions
   - Interference pattern strengthens
   - Centroid begins to shift

3. **t = T/2** (Middle): Discontinuity at beam center
   - Maximum interference effects
   - Beam may split or show double peaks
   - Largest beam width distortion

4. **t = 3T/4**: Discontinuity 3/4 through beam
   - Interference fading
   - Beam beginning to recover symmetry

5. **t = T** (Exit): Discontinuity exits at right edge
   - Beam returns to symmetric profile
   - Phase now uniform (but shifted by Δφ)

### Effect of Chirp Rate

Higher chirp rate (α ↑):
- Larger frequency gradient across beam
- Stronger spatial chirping effects
- More pronounced interference patterns
- Larger focal plane spot displacement

### Effect of Phase Discontinuity

Larger Δφ:
- Stronger phase contrast between regions
- Higher fringe visibility
- More dramatic beam distortion
- Larger transient effects

## Output Files

After running simulations, you'll find:

```
aod_snapshot_0.png - t = 0 (discontinuity enters)
aod_snapshot_1.png - t = T/4
aod_snapshot_2.png - t = T/2 (mid-transit)
aod_snapshot_3.png - t = 3T/4
aod_snapshot_4.png - t = T (discontinuity exits)

aod_cross_sections.png - 1D intensity profiles

chirp_comparison_X.png - Individual chirp rate results
chirp_rate_comparison.png - Side-by-side comparison

phase_comparison_X.png - Individual phase discontinuity results
phase_discontinuity_comparison.png - Side-by-side comparison
```

## Technical Notes

### Integration Method
The simulation uses FFT-based integration which:
- Is ~100× faster than adaptive quadrature
- Maintains good accuracy for smooth integrands
- Requires careful grid sizing for convergence

### Coordinate Systems
- **Crystal coordinates**: (ξ, η) - position in AOD
- **Focal plane coordinates**: (u, v) - position after lens
- **Relationship**: u = λF·k_x where k_x is spatial frequency

### Convergence Parameters
- `n_spatial_points`: Output grid resolution (150 typical)
- `n_integration_points`: Integration grid (200 typical)
- Increase for higher accuracy, decrease for speed

## Validation

The code includes several consistency checks:
1. Beam width in v-direction (unaffected by discontinuity) remains constant
2. Total power conservation (within numerical error)
3. Symmetry recovery after discontinuity exits
4. Limiting cases (α→0 or Δφ→0) return to unperturbed beam

## Animation Scripts

### Single-Tone Animation (`create_animation_fixed_zoom.py`)

Creates a high-quality 4-panel animation of the beam evolution:

```bash
python create_animation_fixed_zoom.py
```

**Features:**
- **Panel 1**: Frequency distribution in crystal with circular beam waist indicator
- **Panel 2**: 1.5mm × 1.5mm zoom on focal plane, centered on beam at t=0
- **Panel 3**: Horizontal cross-section showing intensity profile evolution
- **Panel 4**: Wide field of view showing full focal plane

**Output:** `beam_animation_fixed_zoom.gif`
- 100 frames covering 0 to 1.5×T (0.692 μs)
- Normalized intensity (max = 1.0 at t=0)
- 800×800 FFT grid for extended focal plane coverage
- Shows beam disappearing as it transitions from 80 MHz to 50 MHz position

### Two-Tone Animation (`create_animation_two_tone.py`)

Simulates a two-tone input beam (e.g., from a previous AOD stage):

```bash
python create_animation_two_tone.py
```

**Physics:**
- Input has two frequency components separated by 30 MHz
- Creates three output beams:
  - **50 MHz**: Component 1 in NEW region only
  - **80 MHz**: Component 1 OLD + Component 2 NEW (overlapping) → **Interference!**
  - **110 MHz**: Component 2 in OLD region only
- Middle beam shows dramatic interference patterns

**Output:** `beam_animation_two_tone.gif`
- 100 frames for smooth animation
- Wide FOV panel zoomed to 10-30mm showing all 3 beams
- ~7mm spacing between adjacent beams (matches theory: 0.24 mm/MHz × 30 MHz)
- Clear visualization of beam splitting and interference

**Key Features:**
- Exact FFT values (no interpolation) - boolean masking extracts exact grid points
- Circular beam waist overlay in crystal panel
- All intensities normalized to peak at t=0
- Fixed zoom centered on actual beam position

## Future Extensions

Possible additions:
1. ~~Animation generation (video of temporal evolution)~~ ✓ Implemented
2. Multiple discontinuities (burst mode)
3. Non-Gaussian beam profiles
4. Aberration effects
5. Anisotropic diffraction
6. Temperature effects on V
7. N-tone input configurations
8. Real-time parameter adjustment

## References

- Acousto-Optic theory: Yariv & Yeh, "Optical Waves in Crystals"
- Fourier optics: Goodman, "Introduction to Fourier Optics"
- TeO₂ properties: Dixon, "Acoustic diffraction of light in anisotropic media"

## Authors

Simulation developed for studying transient effects in chirped AOD systems.

## License

Academic/research use.

# %%
import numpy as np
import matplotlib.pyplot as plt
import shutil

from src.material import Material
from src.mesh2D import Mesh
from src.particle import ParticleSet
from src.lagrange_basis import lagrange_basis_Q4
from src.quadrature import gauss_2D
from src.plot_utils import plot_mpm_domain
from src.solver import run_mpm_solver, NodeState, build_particle_element_map
from src.vtk_export import write_pvd

np.set_printoptions(precision=3, suppress=True)


def print_section(title):
    divider = "=" * 70
    print(f"\n{divider}\n{title}\n{divider}")


# ----- Material — Aluminium 2024-T3 -----
rho       = 2700.0
E         = 73.1e9
nu        = 0.33
A         = 369e6
B         = 684e6
n         = 0.73
C         = 0.0083
eps_dot_0 = 1.0
D1, D2, D3, D4 = 0.13, 0.13, -1.5, 0.011

material = Material(rho, E, nu, A, B, n, C, eps_dot_0, D1=D1, D2=D2, D3=D3, D4=D4)

print_section("Material Properties")
print(f"E = {E:.2e} Pa,  nu = {nu},  rho = {rho} kg/m³")
print(f"P-wave speed: {material.wave_speed:.2f} m/s")

# ----- Bar geometry & impact velocity -----
D_bar = 0.0064   # width  (m) — 6.4 mm
L_bar = 0.0324   # height (m) — 32.4 mm, L/D ≈ 5
V0    = 200.0    # impact speed (m/s), downward

# ----- Computational grid -----
# Domain wider than bar to give mushrooming room; taller than bar for free tip travel
Lx, Ly = 0.020, 0.050   # 20 mm × 50 mm
numx, numy = 20, 50      # 1 mm cells

mesh = Mesh(Lx, Ly, numx, numy)
print_section("Computational Grid")
print(f"Domain: {Lx*1e3:.0f} mm × {Ly*1e3:.0f} mm")
print(f"Grid:   {numx} × {numy}  ({mesh.deltax*1e3:.1f} mm cells)")

# ----- Particle mesh -----
# Bar bottom at y=0 — in contact with the rigid wall (bNodes) from t=0
noX, noY = 6, 30                        # 6×30 elements across D×L
pmesh = Mesh(D_bar, L_bar, noX, noY)
pmesh.node[:, 0] += Lx/2 - D_bar/2     # centre bar horizontally in domain

print_section("Bar Geometry")
print(f"D = {D_bar*1e3:.1f} mm,  L = {L_bar*1e3:.1f} mm  (L/D = {L_bar/D_bar:.1f})")
print(f"Particle mesh: {noX} × {noY} elements")

# ----- Particle initialisation -----
ngp    = 2                              # 2×2 Gauss points per element
W, Q   = gauss_2D(ngp)
pCount = len(pmesh.element) * len(W)

particles = ParticleSet(pCount)

pid = 0
for e in range(len(pmesh.element)):
    sctr = pmesh.element[e, :]
    pts  = pmesh.node[sctr, :]
    for q in range(len(W)):
        N, dNdxi     = lagrange_basis_Q4(Q[q])
        J0           = dNdxi.T @ pts
        a            = W[q] * np.linalg.det(J0)
        particles.volume[pid]               = a
        particles.mass[pid]                 = a * rho
        particles.positions[pid]            = N.T @ pts
        particles.deformation_gradient[pid] = [1.0, 0.0, 0.0, 1.0]
        particles.velocities[pid]           = [0.0, -V0]   # all particles hit wall at V0
        pid += 1

particles.set_initial_state()
# No dirichlet_particles or neumann_particles needed:
#   rigid wall = bNodes (y=0), zeroed automatically by the solver every step.

print_section("Particle Initialisation")
print(f"Particles:    {particles.count}")
print(f"Total mass:   {np.sum(particles.mass)*1e3:.3f} g")
print(f"Total volume: {np.sum(particles.volume)*1e6:.3f} cm²")

# ----- Visualise initial configuration -----
print_section("Visualisation — Initial Configuration")
fig, ax = plot_mpm_domain(
    mesh.node, mesh.element, particles.positions,
    figsize=(6, 10), particle_y_scale=1.0,
    particle_y_ref=particles.initial_positions,
)
ax.set_title(f'Taylor Bar — Initial  (V₀ = {V0} m/s)', fontsize=13, fontweight='bold')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
plt.tight_layout(); plt.show()

# ----- Element-particle mapping -----
particles.pElems, particles.mpoints = build_particle_element_map(particles.positions, mesh)

print_section("Element-Particle Mapping")
print(f"Particles located:        {sum(len(m) for m in particles.mpoints)}")
print(f"Elements with particles:  {sum(1 for m in particles.mpoints if m)} / {mesh.elemCount}")

# ----- Solver setup -----
node_state = NodeState(mesh.nodeCount)

dtcrit = mesh.deltax / material.wave_speed
dtime  = 0.5 * dtcrit
nsteps = 1200                   # ~95 μs physical time (well past full mushrooming)
time   = nsteps * dtime

print_section("Solver")
print(f"CFL dt:      {dtcrit:.3e} s")
print(f"Using dt:    {dtime:.3e} s  (CFL × 0.5)")
print(f"Total time:  {time*1e6:.1f} μs  ({nsteps} steps)")
print(f"Impact V₀:   {V0} m/s  →  bar traversal time: {L_bar/material.wave_speed*1e6:.1f} μs")

shutil.rmtree('vtk_output', ignore_errors=True)

# %%
solver_results = run_mpm_solver(
    mesh=mesh,
    particles=particles,
    material=material,
    traction=0.0,
    g=0.0,          # gravity negligible compared to 200 m/s impact forces
    dtime=dtime,
    time=time,
    alpha=0.99,
    node_state=node_state,
    vtk_output_dir='vtk_output',
    vtk_interval=10,
)

print_section("Solver Complete")
pvd_path = write_pvd('vtk_output', solver_results['vtk_entries'])
print(f"VTK steps saved: {len(solver_results['vtk_entries'])}  →  {pvd_path}")

# ----- Post-processing: benchmark metrics -----
print_section("Benchmark Metrics")
y_final = particles.positions[:, 1]
x_final = particles.positions[:, 0]

L_final   = y_final.max() - y_final.min()
D_tip     = x_final.max() - x_final.min()           # mushroom diameter at impact end
L_initial = L_bar

print(f"Initial length:    {L_initial*1e3:.2f} mm")
print(f"Final length:      {L_final*1e3:.2f} mm  (compression: {(1 - L_final/L_initial)*100:.1f}%)")
print(f"Mushroom diameter: {D_tip*1e3:.2f} mm  (initial: {D_bar*1e3:.1f} mm)")

# %%
# ----- Visualise final configuration -----
print_section("Visualisation — Final Configuration")
fig, ax = plot_mpm_domain(
    mesh.node, mesh.element, particles.positions,
    figsize=(6, 10), particle_y_scale=1.0,
    particle_y_ref=particles.initial_positions,
)
ax.set_title('Taylor Bar — Final Deformed Shape', fontsize=13, fontweight='bold')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
plt.tight_layout(); plt.show()

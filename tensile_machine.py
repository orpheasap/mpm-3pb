# %%
import numpy as np
import matplotlib.pyplot as plt
import shutil

from src.material import Material
from src.mesh2D import Mesh
from src.particle import ParticleSet
from src.lagrange_basis import lagrange_basis_Q4
from src.quadrature import gauss_2D
from src.plot_utils import plot_mpm_domain, plot_particles
from src.solver_tensile_machine import run_mpm_solver, NodeState, build_particle_element_map
from src.vtk_export import write_pvd


np.set_printoptions(precision=3, suppress=True)


def print_section(title):
    divider = "=" * 70
    print(f"\n{divider}\n{title}\n{divider}")


def print_subsection(title):
    divider = "-" * 70
    print(f"\n{divider}\n{title}\n{divider}")


# ----- Material Properties — Gelatin (Jell-O approximation) -----
rho  = 1020.0   # density           (kg/m³)  — close to water
E    = 5e3      # Young's modulus   (Pa)     — ~5 kPa, very soft gel
nu   = 0.48     # Poisson's ratio            — nearly incompressible

# Johnson-Cook flow stress: sigma_f = [A + B*eps_p^n][1 + C*ln(eps_dot*)]
A         = 300.0  # initial yield stress    (Pa)  — yields at ~300 Pa
B         = 500.0  # hardening modulus       (Pa)
n         = 0.5    # hardening exponent
C         = 0.05   # strain-rate sensitivity        — gels are more rate-sensitive than metals
eps_dot_0 = 1.0    # reference strain rate   (1/s)

# Johnson-Cook damage: eps_f = [D1 + D2*exp(D3*sigma*)][1 + D4*ln(eps_dot*)]
D1 =  0.30   # fracture strain offset             — fractures at ~30-50% strain
D2 =  0.40   # fracture strain scale
D3 = -1.5    # triaxiality sensitivity
D4 =  0.05   # strain-rate sensitivity of damage

material = Material(rho, E, nu, A, B, n, C, eps_dot_0, D1=D1, D2=D2, D3=D3, D4=D4)

print_section("Material Properties")
print(f"E = {E:.2e} Pa,  nu = {nu},  rho = {rho} kg/m³")
print(f"G = {material.G:.2e} Pa,  K = {material.K:.2e} Pa")
print(f"JC: A={A:.2e}, B={B:.2e}, n={n}, C={C}, eps_dot_0={eps_dot_0}")
print(f"P-wave speed: {material.wave_speed:.2f} m/s")

# ----- Computational grid -----
Lx   = 3
Ly   = 3.2
numx = 24
numy = 24

mesh = Mesh(Lx, Ly, numx, numy)
print_section("Computational Grid")
print(f"Domain size: Lx = {Lx}, Ly = {Ly}")
print(f"Grid resolution: {numx} x {numy} elements")
print(f"Total nodes: {mesh.nodeCount},  Total elements: {mesh.elemCount}")
print(f"Cell size: dx = {mesh.deltax:.4f}, dy = {mesh.deltay:.4f}")

# ----- Particle distribution -----
Lxp = .5
Lyp = 3
noX = 4
noY = 16

pmesh = Mesh(Lxp, Lyp, noX, noY)
pmesh.node[:, 0] += Lx/2 - Lxp/2   # centre beam horizontaly in domain

print_section("Particle Mesh")
print(f"Domain size: Lxp = {Lxp}, Lyp = {Lyp}")
print(f"Grid resolution: {noX} x {noY} elements")

# ----- Particle initialisation -----
ngp    = 5
W, Q   = gauss_2D(ngp)
pCount = len(pmesh.element) * len(W)

particles = ParticleSet(pCount)

pid = 0
max_p_height = 0.0
for e in range(len(pmesh.element)):
    sctr = pmesh.element[e, :]
    pts  = pmesh.node[sctr, :]
    for q in range(len(W)):
        N, dNdxi = lagrange_basis_Q4(Q[q])
        J0    = dNdxi.T @ pts
        detJ0 = np.linalg.det(J0)
        a     = W[q] * detJ0
        particles.volume[pid]               = a
        particles.mass[pid]                 = a * rho
        particles.positions[pid]            = N.T @ pts
        particles.deformation_gradient[pid] = [1.0, 0.0, 0.0, 1.0]
        if particles.positions[pid, 1] > max_p_height:
            max_p_height = particles.positions[pid, 1]
        pid += 1

particles.set_initial_state()

# Dirichlet BC: top particles being pulled
for p in range(particles.count):
    x = particles.positions[p, 0]
    if abs(particles.positions[p, 1] - max_p_height) < 1e-6:
        particles.neumann_particles[p] = True

print_section("Particle Initialisation")
print(f"Number of particles: {particles.count}")
print(f"Total mass:   {np.sum(particles.mass):.4f} kg")
print(f"Total volume: {np.sum(particles.volume):.4f} m²")
print(f"Neumann BC particles: {np.sum(particles.neumann_particles)}")

# ----- Visualise initial configuration -----
print_section("Visualisation — Initial Configuration")
fig, ax = plot_mpm_domain(
    mesh.node, mesh.element, particles.positions,
    figsize=(12, 8), particle_y_scale=1.0,
    particle_y_ref=particles.initial_positions,
)
neumann_pos = particles.positions[particles.neumann_particles]
ax.scatter(neumann_pos[:, 0], neumann_pos[:, 1],
           s=32, c='green', marker='o', alpha=0.9, label='Neumann BC')
ax.set_title('Initial Particle Positions', fontsize=14, fontweight='bold')
plt.show()

# ----- Element-particle mapping -----
particles.pElems, particles.mpoints = build_particle_element_map(particles.positions, mesh)

print_section("Element-Particle Mapping")
print(f"Particles located: {sum(len(m) for m in particles.mpoints)}")
print(f"Elements with particles: {sum(1 for m in particles.mpoints if m)} / {mesh.elemCount}")

# ----- Solver setup -----
node_state = NodeState(mesh.nodeCount)

dtcrit = mesh.deltax / material.wave_speed
dtime  = 0.5 * dtcrit      # safety factor of 0.5 on CFL
time   = 2700 * dtime       # total simulation time


print_section("Solver")
print(f"CFL critical dt: {dtcrit:.2e} s")
print(f"Using dt:        {dtime:.2e} s  (safety factor 0.5)")
print(f"Total time:      {time:.2e} s  ({int(time/dtime)} steps)")

shutil.rmtree('vtk_output', ignore_errors=True)

v_pull   = -0.1   # upward grip speed (m/s) — slow pull on soft gel
print(f"total elongation : {time*v_pull:.2e} m")

# %%
solver_results = run_mpm_solver(
    mesh=mesh,
    particles=particles,
    material=material,
    g=0,
    dtime=dtime,
    time=time,
    alpha=0.99,
    node_state=node_state,
    vtk_output_dir='vtk_output',
    vtk_interval=10,
    v_pull=v_pull,
)

print_section("Solver Complete")
print(f"Steps run:       {len(solver_results['time'])}")
print(f"Final KE:        {solver_results['kinetic'][-1]:.4e} J")

pvd_path = write_pvd('vtk_output', solver_results['vtk_entries'])
print(f"VTK steps saved: {len(solver_results['vtk_entries'])}  →  {pvd_path}")

# %%
# ----- Visualise final configuration -----
print_section("Visualisation — Final Configuration")
fig, ax = plot_mpm_domain(
    mesh.node, mesh.element, particles.positions,
    figsize=(12, 8), particle_y_scale=1.0,
    particle_y_ref=particles.initial_positions,
)
ax.set_title('Final Particle Positions', fontsize=14, fontweight='bold')
plt.show()

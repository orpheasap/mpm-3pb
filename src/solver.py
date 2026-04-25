import numpy as np
from src.vtk_export import write_particles_vtp


class NodeState:
    """Container for nodal MPM state variables."""

    def __init__(self, node_count):
        self.node_count = node_count
        self.mass = np.zeros(node_count)
        self.momentum = np.zeros((node_count, 2))
        self.internal_force = np.zeros((node_count, 2))
        self.external_force = np.zeros((node_count, 2))

    def reset(self):
        self.mass.fill(0.0)
        self.momentum.fill(0.0)
        self.internal_force.fill(0.0)
        self.external_force.fill(0.0)


def get_mpm2d_shape(x, deltax, deltay):
    """
    Return the MPM nodal shape function and gradient for a particle
    relative to a grid node at the origin.
    
    The shape function is a tensor product of 1D tent functions.
    """
    xi = x[0] / deltax
    eta = x[1] / deltay

    if abs(xi) >= 1.0 or abs(eta) >= 1.0:
        return 0.0, np.zeros(2)

    Nx = 1.0 - abs(xi)
    Ny = 1.0 - abs(eta)
    N = Nx * Ny

    # gradient dN/dx, dN/dy
    dNdx = np.zeros(2)
    dNdx[0] = -(np.sign(xi) if xi != 0 else 0.0) * Ny / deltax
    dNdx[1] = -(np.sign(eta) if eta != 0 else 0.0) * Nx / deltay

    return N, dNdx


def build_particle_element_map(xp, mesh):
    """
    Build a mapping from particles to element indices and
    from elements to particle indices.
    """
    pElems = np.zeros(len(xp), dtype=int)
    mpoints = [[] for _ in range(mesh.elemCount)]

    for p, point in enumerate(xp):
        ix = int(np.floor(point[0] / mesh.deltax))
        iy = int(np.floor(point[1] / mesh.deltay))
        ix = max(0, min(ix, mesh.numx - 1)) # clamp to valid range
        iy = max(0, min(iy, mesh.numy - 1)) # clamp to valid range
        e = ix + iy * mesh.numx
        pElems[p] = e
        mpoints[e].append(p)

    return pElems, mpoints


def run_mpm_solver(mesh, particles, material, traction,
                   g=9.81, dtime=1e-5, time=1e-2, tol=1e-4,
                   node_state=None, vtk_output_dir=None, vtk_interval=10):
    """
    Run a simple MPM solver using Algorithm 1 (PIC/FLIP-style).

    Parameters:
    mesh:      Mesh object
    particles: ParticleSet object
    material:  Material object
    traction:  Applied traction for Neumann BC (N/m)
    g:         Gravity magnitude (positive downward, m/s^2)
    dtime:     Time step (s)
    time:      Total simulation time (s)
    tol:       Zero-mass tolerance

    Returns:
    dict containing energy histories and final state
    """
    node_state = node_state or NodeState(mesh.nodeCount) 
    if node_state.node_count != mesh.nodeCount:
        raise ValueError("NodeState size must match mesh.nodeCount")

    _, mpoints = build_particle_element_map(particles.positions, mesh)

    ta = []
    ka = []
    sa = []
    vtk_entries = []

    I = np.eye(2)
    nsteps = int(np.floor(time / dtime))
    t = 0.0

    for istep in range(nsteps):
        node_state.reset()

        for e in range(mesh.elemCount):
            esctr = mesh.element[e]
            for pid in mpoints[e]:  # P2G
                stress = particles.stress[pid]
                for idn in esctr:
                    x = particles.positions[pid] - mesh.node[idn]
                    N, dNdx = get_mpm2d_shape(x, mesh.deltax, mesh.deltay)

                    node_state.mass[idn]          += N * particles.mass[pid]
                    node_state.momentum[idn]      += N * particles.mass[pid] * particles.velocities[pid]
                    node_state.internal_force[idn, 0] -= particles.volume[pid] * (stress[0] * dNdx[0] + stress[2] * dNdx[1])
                    node_state.internal_force[idn, 1] -= particles.volume[pid] * (stress[2] * dNdx[0] + stress[1] * dNdx[1])
                    node_state.external_force[idn, 1] -= g * N * particles.mass[pid]
                    if particles.neumann_particles[pid] and t < 0.01 * time:
                        node_state.external_force[idn, 1] -= traction * N * particles.volume[pid]

        node_state.momentum += (node_state.internal_force + node_state.external_force) * dtime

        if mesh.lNodes.size > 0:
            node_state.momentum[mesh.lNodes, :] = 0.0
        if mesh.rNodes.size > 0:
            node_state.momentum[mesh.rNodes, :] = 0.0
        if mesh.bNodes.size > 0:
            node_state.momentum[mesh.bNodes, :] = 0.0
        if mesh.tNodes.size > 0:
            node_state.momentum[mesh.tNodes, :] = 0.0

        k = 0.0
        u = 0.0

        for e in range(mesh.elemCount):  # G2P
            for pid in mpoints[e]:
                Lp = np.zeros((2, 2))
                for idn in mesh.element[e]:
                    x = particles.positions[pid] - mesh.node[idn]
                    N, dNdx = get_mpm2d_shape(x, mesh.deltax, mesh.deltay)

                    vI = np.zeros(2)
                    if node_state.mass[idn] > tol:
                        particles.velocities[pid] += dtime * N * (node_state.internal_force[idn] + node_state.external_force[idn]) / node_state.mass[idn]
                        particles.positions[pid]  += dtime * N * node_state.momentum[idn] / node_state.mass[idn]
                        vI = node_state.momentum[idn] / node_state.mass[idn]

                    Lp += np.outer(vI, dNdx)

                F = (I + Lp * dtime) @ particles.deformation_gradient[pid].reshape(2, 2)
                particles.deformation_gradient[pid] = F.reshape(4)
                particles.volume[pid] = np.linalg.det(F) * particles.initial_volume[pid]
                dEps = 0.5 * dtime * (Lp + Lp.T)
                dEps_vec = np.array([dEps[0, 0], dEps[1, 1], 2.0 * dEps[0, 1]])
                particles.stress[pid]  += material.elasticity_matrix @ dEps_vec
                particles.strain[pid]  += dEps_vec

                k += 0.5 * (particles.velocities[pid, 0]**2 + particles.velocities[pid, 1]**2) * particles.mass[pid]
                u += 0.5 * particles.volume[pid] * particles.stress[pid] @ particles.strain[pid]

        ta.append(t)
        ka.append(k)
        sa.append(u)

        if vtk_output_dir is not None and istep % vtk_interval == 0:
            fname = write_particles_vtp(
                particles.positions, particles.velocities,
                particles.stress, particles.strain,
                istep, t, vtk_output_dir,
            )
            vtk_entries.append((t, fname))

        _, mpoints = build_particle_element_map(particles.positions, mesh)

        t += dtime

    return {
        'time': np.array(ta),
        'kinetic': np.array(ka),
        'strain': np.array(sa),
        'vtk_entries': vtk_entries,
    }

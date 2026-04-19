import numpy as np


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


def run_mpm_solver(mesh, particles, traction, xp, vp, Mp, Vp, Fp, s, eps, Vp0, rho, C,
                   g=9.81, dtime=1e-5, time=1e-2, tol=1e-4,
                   node_state=None):
    """
    Run a simple MPM solver using Algorithm 1 (PIC/FLIP-style).

    Parameters:
    mesh: Mesh object with node, element, deltax, deltay, numx, numy, elemCount, nodeCount
    particles: ParticleSet object
    traction: Applied traction for Neumann BC (N/m)
    xp: Particle positions (pCount, 2)
    vp: Particle velocities (pCount, 2)
    Mp: Particle masses (pCount,)
    Vp: Particle volumes (pCount,)
    Fp: Particle deformation gradients (pCount, 4)
    s: Particle stress vectors (pCount, 3)
    eps: Particle strains (pCount, 3)
    Vp0: Initial particle volumes (pCount,)
    rho: Density
    C: Elasticity matrix (3x3)
    g: gravity acceleration magnitude (positive downward)
    dtime: time step
    time: final time
    tol: zero-mass tolerance
    left_nodes, right_nodes: arrays of node indices for Dirichlet BCs

    Returns:
    result: dict containing histories and final state arrays
    """
    elemCount = mesh.elemCount
    nodeCount = mesh.nodeCount
    pCount = len(xp)

    node_state = node_state or NodeState(nodeCount) # create if not provided
    if node_state.node_count != nodeCount: # sanity check
        raise ValueError("NodeState size must match mesh.nodeCount")

    pElems, mpoints = build_particle_element_map(xp, mesh) # initial mapping

    pos_history = []
    vel_history = []
    ta = []
    ka = []
    sa = []

    I = np.eye(2)
    nsteps = int(np.floor(time / dtime))
    t = 0.0

    
    for istep in range(nsteps): # main time-stepping loop
        node_state.reset()

        for e in range(elemCount):
            esctr = mesh.element[e]
            mpts = mpoints[e]

            for pid in mpts: # P2G loop
                stress = s[pid]

                for idn in esctr: # loop over nodes of this element
                    x = xp[pid] - mesh.node[idn]
                    N, dNdx = get_mpm2d_shape(x, mesh.deltax, mesh.deltay)

                    node_state.mass[idn] += N * Mp[pid] # mass contribution from particle to node
                    node_state.momentum[idn] += N * Mp[pid] * vp[pid] # momentum contribution from particle to node

                    node_state.internal_force[idn, 0] -= Vp[pid] * (stress[0] * dNdx[0] + stress[2] * dNdx[1]) # internal force contribution in x
                    node_state.internal_force[idn, 1] -= Vp[pid] * (stress[2] * dNdx[0] + stress[1] * dNdx[1]) # internal force contribution in y
                    node_state.external_force[idn, 1] -= g * N * Mp[pid] # gravity force contribution in y (newmann BC)
                    # apply Neumann BC if this node has particles flagged for Neumann BC
                    if particles.neumann_particles[pid]: # check if this particle is in the Neumann BC region
                        node_state.external_force[idn, 1] -= traction * N * Vp[pid] # example downward force for Neumann BC
       
        node_state.momentum += (node_state.internal_force + node_state.external_force) * dtime # update nodal momentum

        if mesh.lNodes.size > 0: # zero out momentum for left boundary nodes (Dirichlet BC)
            node_state.momentum[mesh.lNodes, :] = 0.0
        if mesh.rNodes.size > 0: # zero out momentum for right boundary nodes (Dirichlet BC)
            node_state.momentum[mesh.rNodes, :] = 0.0

        k = 0.0
        u = 0.0

        for e in range(elemCount): # G2P loop
            mpts = mpoints[e]

            for pid in mpts:
                Lp = np.zeros((2, 2))

                for idn in mesh.element[e]:
                    x = xp[pid] - mesh.node[idn] # relative position of particle to node
                    N, dNdx = get_mpm2d_shape(x, mesh.deltax, mesh.deltay) # shape function and gradient for this particle-node pair

                    vI = np.zeros(2)
                    if node_state.mass[idn] > tol: # avoid division by zero
                        vp[pid] += dtime * N * (node_state.internal_force[idn] + node_state.external_force[idn]) / node_state.mass[idn] # update particle velocity from nodal forces
                        xp[pid] += dtime * N * node_state.momentum[idn] / node_state.mass[idn] # update particle position from nodal momentum
                        vI = node_state.momentum[idn] / node_state.mass[idn] # nodal velocity for this node

                    Lp += np.outer(vI, dNdx) # velocity gradient contribution from this node

                F = (I + Lp * dtime) @ Fp[pid].reshape(2, 2) # update deformation gradient
                Fp[pid] = F.reshape(4) # store updated deformation gradient
                Vp[pid] = np.linalg.det(F) * Vp0[pid] # update particle volume from deformation gradient
                dEps = 0.5 * dtime * (Lp + Lp.T) # strain increment from velocity gradient
                dsigma = C @ np.array([dEps[0, 0], dEps[1, 1], 2.0 * dEps[0, 1]]) # stress increment from strain increment using elasticity matrix
                s[pid] += dsigma # update particle stress
                eps[pid] += np.array([dEps[0, 0], dEps[1, 1], 2.0 * dEps[0, 1]]) # update particle strain
                
                k += 0.5 * (vp[pid, 0]**2 + vp[pid, 1]**2) * Mp[pid] # kinetic energy contribution from this particle
                u += 0.5 * Vp[pid] * s[pid] @ eps[pid] # strain energy contribution from this particle

        pos_history.append(xp.copy()) # store particle positions for this time step
        vel_history.append(vp.copy()) # store particle velocities for this time step
        ta.append(t) # store time for this time step
        ka.append(k) # store kinetic energy for this time step
        sa.append(u) # store strain energy for this time step

        pElems, mpoints = build_particle_element_map(xp, mesh) # update particle-element mapping for next step

        t += dtime # increment time

    return {
        'time': np.array(ta),
        'kinetic': np.array(ka),
        'strain': np.array(sa),
        'pos': pos_history,
        'vel': vel_history,
        'pElems': pElems,
        'mpoints': mpoints,
        'xp': xp,
        'vp': vp,
        'Mp': Mp,
        'Vp': Vp,
        'Fp': Fp,
        's': s,
        'eps': eps,
        'node_state': node_state,
    }

import numpy as np

class ParticleSet:
    """Simple container for particle state arrays."""

    def __init__(self, count):
        self.count = count
        self.positions = np.zeros((count, 2), dtype=float)
        self.velocities = np.zeros((count, 2), dtype=float)
        self.mass = np.ones(count, dtype=float)
        self.volume = np.ones(count, dtype=float)
        self.deformation_gradient = np.tile(np.array([1.0, 0.0, 0.0, 1.0], dtype=float), (count, 1))
        self.stress = np.zeros((count, 3), dtype=float)
        self.strain = np.zeros((count, 3), dtype=float)
        self.initial_positions = np.zeros((count, 2), dtype=float)
        self.initial_volume = np.ones(count, dtype=float)
        self.pElems = np.zeros(count, dtype=int) # particle to element mapping
        self.mpoints = [[] for _ in range(0)]
        self.neumann_particles = np.zeros(count, dtype=bool) # flag for Neumann BC particles

    def set_initial_state(self):
        self.initial_positions = self.positions.copy()
        self.initial_volume = self.volume.copy()

    def displacement(self):
        return self.positions - self.initial_positions

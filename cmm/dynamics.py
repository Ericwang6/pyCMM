import torch
from .units import FS2AU

class VelocityVerlet:
    """
    Velocity Verlet integrator implemented in PyTorch to run on GPU.
    
    This integrator uses the velocity Verlet algorithm to propagate a system
    of particles in time, with positions, velocities, and forces updated
    according to the force field provided.
    """
    def __init__(
            self,
            velocities: torch.Tensor,
            masses: torch.Tensor,
            timestep=0.5 * FS2AU,
            device='cuda'
        ):
        """
        Initialize the Velocity Verlet integrator.
        
        Args:
            positions (torch.Tensor): Initial positions of shape (n_atoms, 3)
            velocities (torch.Tensor): Initial velocities of shape (n_atoms, 3)
            masses (torch.Tensor): Masses of shape (n_atoms,) or (n_atoms, 1)
            force_fn (callable): Function that takes positions and returns forces
            timestep (float): Simulation timestep
            device (str): Device to run the simulation on ('cuda' or 'cpu')
        """
        self.device = device
        
        self.velocities = velocities
        self.masses = masses
        
        # Reshape masses for proper broadcasting if needed
        if len(self.masses.shape) == 1:
            self.masses = self.masses.view(-1, 1)
        
        self.last_forces = torch.zeros_like(self.velocities)
        self.timestep = timestep
        
        # For tracking
        self.step_count = 0
        self.time = 0.0

    def first_half_step(self, positions: torch.Tensor):
        """
        Perform a Velocity Verlet integration half-step.
        
        Returns:
            Updated positions
        """
        
        # Half-step velocity update
        self.velocities += 0.5 * self.timestep * self.last_forces / self.masses
        self.last_forces = -positions.grad.detach().clone()
        
        # Full position update
        return positions + self.timestep * self.velocities
    
    def second_half_step(self):
        """
        Perform a Velocity Verlet integration half-step.
        
        Returns:
            Updated positions
        """
        
        # Half-step velocity update
        self.velocities += 0.5 * self.timestep * self.last_forces / self.masses
        
        # Update tracking variables
        self.step_count += 1
        self.time += self.timestep
"""
Implementation of the Particle Swarm Optimization (PSO) algorithm.

This module contains the base implementation of the PSO algorithm that will be used
to optimize image features that generate higher arousal and valence.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Particle:
    """
    Represents a particle in the PSO algorithm.

    Attributes:
        position: Current position of the particle in the search space
        velocity: Current velocity of the particle
        best_position: Best position found by the particle
        best_score: Best score found by the particle
    """

    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_score: float


class PSOOptimizer:
    """
    Implementation of the Particle Swarm Optimization algorithm.

    This optimizer seeks to find the best combination of features
    that maximize arousal and valence in user responses.
    """

    def __init__(
        self,
        n_particles: int,
        n_dimensions: int,
        objective_function: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        w_start: float = 0.9,
        w_end: float = 0.4,
        c1: float = 2.0,
        c2: float = 1.0,
        max_iter: int = 200,
        velocity_clamp: float = 0.5,
        convergence_threshold: float = 1e-6,
        convergence_window: int = 10,
    ):
        """
        Initialize the PSO optimizer.

        Args:
            n_particles: Number of particles in the swarm
            n_dimensions: Number of dimensions in the search space
            objective_function: Objective function to optimize
            bounds: Bounds for each dimension of the search space
            w_start: Initial inertia weight
            w_end: Final inertia weight
            c1: Cognitive learning factor
            c2: Social learning factor
            max_iter: Maximum number of iterations
            velocity_clamp: Maximum velocity magnitude
            convergence_threshold: Threshold for convergence detection
            convergence_window: Number of iterations to check for convergence
        """
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.objective_function = objective_function
        self.bounds = bounds
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.velocity_clamp = velocity_clamp
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window

        self.particles: list[Particle] = []
        self.global_best_position = None
        self.global_best_score = float("-inf")
        self.convergence_curve = []
        self.best_scores_history = []

        self._initialize_particles()

    def _initialize_particles(self):
        """
        Initialize particles with random positions and velocities.

        This method creates the initial swarm of particles with random positions
        within the specified bounds and zero initial velocities.
        """
        for _ in range(self.n_particles):
            position = np.array(
                [np.random.uniform(low, high) for low, high in self.bounds]
            )
            velocity = np.zeros(self.n_dimensions)
            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                best_score=float("-inf"),
            )
            self.particles.append(particle)

    def _check_convergence(self) -> bool:
        """
        Check if the optimization has converged.

        Returns:
            bool: True if converged, False otherwise
        """
        if len(self.best_scores_history) < self.convergence_window:
            return False

        recent_scores = self.best_scores_history[-self.convergence_window :]
        return np.std(recent_scores) < self.convergence_threshold

    def optimize(self) -> tuple[np.ndarray, float]:
        """
        Run the PSO algorithm.

        Returns:
            tuple[np.ndarray, float]: Best position found and its score
        """
        for iteration in range(self.max_iter):
            # Calculate current inertia weight (linear decrease)
            w = self.w_start - (self.w_start - self.w_end) * (iteration / self.max_iter)

            for particle in self.particles:
                # Evaluate current position
                score = self.objective_function(particle.position)

                # Update personal best position
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()

                # Update global best position
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle.position.copy()

                # Update velocity and position
                r1, r2 = np.random.rand(2)
                particle.velocity = (
                    w * particle.velocity
                    + self.c1 * r1 * (particle.best_position - particle.position)
                    + self.c2 * r2 * (self.global_best_position - particle.position)
                )

                # Apply velocity clamping
                particle.velocity = np.clip(
                    particle.velocity, -self.velocity_clamp, self.velocity_clamp
                )

                particle.position += particle.velocity

                # Apply bounds
                for i, (low, high) in enumerate(self.bounds):
                    particle.position[i] = np.clip(particle.position[i], low, high)

            # Track convergence
            self.best_scores_history.append(self.global_best_score)
            self.convergence_curve.append(float(self.global_best_score))

            # Check for convergence
            if self._check_convergence():
                print(f"Converged at iteration {iteration + 1}")
                break

        return self.global_best_position, self.global_best_score

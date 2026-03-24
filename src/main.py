import numpy as np

class SwarmOptimizer:
    def __init__(self, objective_function, num_particles, num_dimensions, bounds):
        self.objective_function = objective_function
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.bounds = bounds
        self.particles = self.initialize_particles()
        self.global_best_position = None
        self.global_best_fitness = float('inf')

    def initialize_particles(self):
        particles = []
        for _ in range(self.num_particles):
            position = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.num_dimensions)
            velocity = np.zeros(self.num_dimensions)
            personal_best_position = position.copy()
            personal_best_fitness = self.objective_function(position)
            particle = {
                'position': position,
                'velocity': velocity,
                'personal_best_position': personal_best_position,
                'personal_best_fitness': personal_best_fitness
            }
            particles.append(particle)
        return particles

    def update_particle(self, particle, w, c1, c2):
        r1 = np.random.uniform(0, 1, self.num_dimensions)
        r2 = np.random.uniform(0, 1, self.num_dimensions)

        particle['velocity'] = w * particle['velocity'] + \\
                               c1 * r1 * (particle['personal_best_position'] - particle['position']) + \\
                               c2 * r2 * (self.global_best_position - particle['position'])
        particle['position'] = particle['position'] + particle['velocity']
        particle['position'] = np.clip(particle['position'], self.bounds[:, 0], self.bounds[:, 1])

        fitness = self.objective_function(particle['position'])
        if fitness < particle['personal_best_fitness']:
            particle['personal_best_position'] = particle['position'].copy()
            particle['personal_best_fitness'] = fitness

        if fitness < self.global_best_fitness:
            self.global_best_position = particle['position'].copy()
            self.global_best_fitness = fitness

    def optimize(self, max_iterations, w, c1, c2):
        for _ in range(max_iterations):
            for particle in self.particles:
                self.update_particle(particle, w, c1, c2)
        return self.global_best_position, self.global_best_fitness

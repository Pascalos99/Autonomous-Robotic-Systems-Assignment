import numpy as np
import matplotlib.pyplot as plt


class Swarm:
    n_particles = None
    value_range = None
    
    particles = None
    a, b, c = None, None, None
    
    def __init__(self, n_particles, value_range):
        self.n_particles = n_particles
        self.value_range = value_range
            
            
    def run(self, title, function, a, b, c, iterations = 1000):
        # Get the specific a, b and c values for each iteration.
        a = np.linspace(a[0], a[1], iterations)
        b = np.linspace(b[0], b[1], iterations)
        c = np.linspace(c[0], c[1], iterations)
        
        # Initialize particles.
        self.particles = []
        for _ in range(self.n_particles):
            # Generate a random initial position and velocity.
            position = np.array((np.random.uniform(self.value_range[0], self.value_range[1]), np.random.uniform(self.value_range[0], self.value_range[1])))
            velocity = np.array((np.random.uniform(self.value_range[0], self.value_range[1]), np.random.uniform(self.value_range[0], self.value_range[1]))) * 0.1
            performance = None
            
            # Add the particle to the swarm.
            particle = Particle(position, velocity, performance)   
            self.particles.append(particle) 
        
        # Set performance for current particle locations.
        for particle in self.particles:
            particle.performance = function(particle.position[0], particle.position[1])
            particle.best_performance = particle.performance
            
        # Get the best particle in the swarm.
        swarm_best_position = min(self.particles, key = lambda particle: particle.best_performance).position
        swarm_best_performance = function(swarm_best_position[0], swarm_best_position[1])
        
        # Enable interactive plotting.
        plt.ion()
        
        # Create the plot.
        fig, ax = plt.subplots()
        
        ax.set_title(f"{ title }: Iteration 0")
        ax.set_xlim(self.value_range)
        ax.set_ylim(self.value_range)
        
        x = np.arange(self.value_range[0], self.value_range[1], 0.05)
        y = np.arange(self.value_range[0], self.value_range[1], 0.05)
        x, y = np.meshgrid(x, y)
        
        ax.contour(x, y, function(x, y), 250)
        
        # Plot the initial situation.
        points = []
        for particle in self.particles:
            points.extend(
                ax.plot(particle.position[0], particle.position[1], 'r*')
            )

        # Iterations of Particle Swarm Optimization.
        for i in range(iterations):
            # Run a iteration of the swarm.
            self.run_iteration(function, a[i], b[i], c[i], 1, swarm_best_position, swarm_best_performance)
            
            # Update the swarm best position and performance.
            swarm_best_position = min(self.particles, key = lambda particle: particle.best_performance).position
            swarm_best_performance = function(swarm_best_position[0], swarm_best_position[1])
            
            # Remove old points from plot.
            for point in points:
                point.remove()
                
            # Plot new points.
            points = []
            for particle in self.particles:
                points.extend(
                    ax.plot(particle.position[0], particle.position[1], 'r*')
                )
                
            ax.set_title(f"{ title }: Iteration { i + 1 }")
                
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        # Disable interactive plotting.
        plt.ioff()
        
        # Show the end result.
        plt.show()
        
    def run_iteration(self, function, a, b, c, dt, swarm_best_position, swarm_best_performance):
        # Update the position and velocity of each particle.
        for particle in self.particles:
            # Update position and then velocity.
            particle.position = particle.position + particle.velocity * dt
            particle.velocity = a * particle.velocity + \
                                b * np.random.uniform(0, 1) * (particle.best_position - particle.position) + \
                                c * np.random.uniform(0, 1) * (swarm_best_position - particle.position)
            
            # Update the performance of the particle.
            particle.performance = function(particle.position[0], particle.position[1])
            if particle.performance < particle.best_performance:
                particle.best_position = particle.position
                particle.best_performance = particle.performance
        
            
class Particle:
    position = None
    velocity = None
    performance = None
    best_position = None
    best_performance = None
    
    def __init__(self, position, velocity, performance):
        self.position = position
        self.velocity = velocity
        self.performance = performance
        self.best_position = position
        self.best_performance = performance


def rosenbrock_benchmark(a = 0, b = 100):
    return lambda x, y: (a - x)**2 + b * (y - x**2)**2


def rastrigin_benchmark():
    return lambda x, y: 20 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))

        
if __name__ == "__main__":
    swarm = Swarm(20, (-3, 3))
    
    print("Running PSO benchmarks...")
    
    # Run the swarm with Rosenbrock benchmark.
    print("Running Rosenbrock benchmark...")
    swarm.run("PSO Rosenbrock", rosenbrock_benchmark(), (0.9, 0.4), (2, 2), (2, 2), 1000)
    
    # Run the swarm with Rastrigin benchmark.
    print("Running Rastrigin benchmark...")
    swarm.run("PSO Rastrigin", rastrigin_benchmark(), (0.9, 0.4), (2, 2), (2, 2), 1000)
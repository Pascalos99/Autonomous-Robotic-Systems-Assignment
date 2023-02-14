import pygame
pygame.init()

WIDTH, HEIGHT = 800, 600
FPS = 60

class Simulation:
    def __init__(self):
        self.win = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        
    def run(self):
        # Main Loop
        # Get Inputs and do Calculations 
        self.is_running = True
        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False

        self.draw()
        self.clock.tick(FPS)
        
    def draw():
        # Draw all changes
        pygame.display.flip()
    
if __name__ == '__main__':
    sim = Simulation()
    sim.run()
    pygame.quit()
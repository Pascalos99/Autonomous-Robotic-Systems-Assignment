import pygame
pygame.init()

WIDTH, HEIGHT = 800, 600
FPS = 60

class Player:
    def __init__(self):
        self.pos = [100, 100]
        
    def move(self, dx, dy):
        self.pos[0] += dx 
        self.pos[1] += dy

class Simulation:
    def __init__(self):
        self.win = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.player = Player()
        
    def run(self):
        # Main Loop
        # Get Inputs and do Calculations 
        self.is_running = True
        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                
                speed = 10
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_o:
                        self.player.move(0, -speed)
                    if event.key == pygame.K_a:
                        self.player.move(-speed, 0)
                    if event.key == pygame.K_s:
                        self.player.move(0, speed)
                    if event.key == pygame.K_d:
                        self.player.move(speed, 0)

            self.draw()
            self.clock.tick(FPS)
        
    def draw(self):
        # Draw all changes
        self.clear()
        pygame.draw.circle(self.win, '#ff0000', self.player.pos, 25)
        pygame.display.flip()
        
    def clear(self):
        self.win.fill('#232323')
    
if __name__ == '__main__':
    sim = Simulation()
    sim.run()
    pygame.quit()
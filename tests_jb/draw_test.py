import pygame
import numpy as np
pygame.init()
pygame.font.init()

WIDTH, HEIGHT = 1200, 800
FPS = 30
FONT = pygame.font.SysFont('Consolas', 14)

class Player:
    def __init__(self):
        self.pos = [WIDTH // 2, HEIGHT // 2]
        self.vel = [0, 0]
        self.size = 35
        self.num_sensors = 12
        self.direction = 0 #degrees, 0 = up, 90 = right, 180 = down, 270 = left
        self.vision_range = 200

    def rotate(self, angle):
        self.direction += angle

    def move(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

    def add_velocity(self, dx, dy):
        self.vel[0] += dx 
        self.vel[1] += dy

class Simulation:
    def __init__(self):
        self.win = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.player = Player()
        
    def run(self):
        self.is_running = True
        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False

                if event.type == pygame.KEYDOWN:
                    # Player movement
                    if event.key == pygame.K_w:
                        self.player.add_velocity(0, -1)
                    if event.key == pygame.K_s:
                        self.player.add_velocity(0, 1)
                    if event.key == pygame.K_a:
                        self.player.add_velocity(-1, 0)
                    if event.key == pygame.K_d:
                        self.player.add_velocity(1, 0)
                    
                    # Player rotation
                    if event.key == pygame.K_e:
                        self.player.rotate(5)
                    elif event.key == pygame.K_q:
                        self.player.rotate(-5)
            
            self.draw()
            self.clock.tick(FPS)
        
    def draw(self):
        self.clear()
        self.player.move()
        self.draw_player()
        pygame.display.flip()

    def draw_player(self):
        # Draw Main Circle
        pygame.draw.circle(self.win, '#00aacc', self.player.pos, self.player.size)
        
        # Draw Direction Line
        pygame.draw.line(self.win, '#000000', self.player.pos, (
            self.player.pos[0] + self.player.size * np.sin(np.deg2rad(self.player.direction)),
            self.player.pos[1] - self.player.size * np.cos(np.deg2rad(self.player.direction)),
        ), width=2)

        # Draw all sensor lines (draw first so player is on top)
        angle = 360 / self.player.num_sensors
        for i in range(self.player.num_sensors):
            pygame.draw.line(self.win, '#dd0000', start_pos=[
                self.player.pos[0] + self.player.size * np.sin(np.deg2rad(i * angle+ self.player.direction)),
                self.player.pos[1] - self.player.size * np.cos(np.deg2rad(i * angle + self.player.direction)),
            ], end_pos=[
                self.player.pos[0] + self.player.vision_range * np.sin(np.deg2rad(i * angle+ self.player.direction)),
                self.player.pos[1] - self.player.vision_range * np.cos(np.deg2rad(i * angle + self.player.direction)),
            ], width=1)
        
        # Show motor numbers
        x_text = FONT.render(f'x:{self.player.vel[0]}', False, '#dddddd')
        y_text = FONT.render(f'y:{self.player.vel[1]}', False, '#dddddd')
        self.win.blit(x_text, dest=[
            self.player.pos[0] - x_text.get_width() // 2 + (self.player.size // 2) * np.sin(np.deg2rad(270+ self.player.direction)),
            self.player.pos[1] - x_text.get_height() // 2 - (self.player.size // 2) * np.cos(np.deg2rad(270 + self.player.direction)),
        ])
        self.win.blit(y_text, dest=[
            self.player.pos[0] - y_text.get_width() // 2 + (self.player.size // 2) * np.sin(np.deg2rad(90 + self.player.direction)),
            self.player.pos[1] - y_text.get_height() // 2 - (self.player.size // 2) * np.cos(np.deg2rad(90 + self.player.direction)),
        ])

        # Show Distance Numbers
        for i in range(self.player.num_sensors):
            text = FONT.render(str(self.player.vision_range), False, '#dddddd')
            self.win.blit(text, dest=[
                self.player.pos[0] - text.get_width() // 2 + (self.player.size + 20) * np.sin(np.deg2rad(i * angle+ self.player.direction)),
                self.player.pos[1] - text.get_height() // 2 - (self.player.size + 20) * np.cos(np.deg2rad(i * angle + self.player.direction)),
            ])
        
    def clear(self):
        self.win.fill('#232323')
    
if __name__ == '__main__':
    sim = Simulation()
    sim.run()
    pygame.quit()
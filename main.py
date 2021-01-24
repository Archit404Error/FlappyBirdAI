import pygame
import os
import math
import sys
import random
import neat

screen_width = 600
screen_height = 800
generation = 0
background = pygame.transform.scale(pygame.image.load("bg.png"), (600, 800))

class Bird:
    def __init__(self):
        self.img = pygame.image.load("bird.png")
        self.img = pygame.transform.scale(self.img, (100, 100))
        self.pos = [100, 400]
        self.accel = 0
        self.center = [self.pos[0] + 50, self.pos[1] + 50]
        self.alive = True
        self.time_alive = 0
        self.jump = -5

    def draw(self, screen):
        screen.blit(self.img, self.pos)

    def get_data(self, pipes):
        index = 0
        if (pipes[0].getPosn()[0] + pipes[0].getDims()[0]) < self.pos[0]:
            index = 2
        downpipe = pipes[index]
        uppipe = pipes[index + 1]
        return [self.pos[0] - downpipe.getPosn()[0], self.pos[1] - downpipe.getPosn()[1], self.pos[0] - uppipe.getPosn()[0], self.pos[1] - uppipe.getPosn()[1]]

    def is_alive(self):
        return self.alive

    def get_reward(self):
        return self.time_alive / 100.0

    def jump_up(self):
        self.jump = 5.5

    def died(self, pipe):
        if(self.pos[1] < 0 or self.pos[1] > 800):
            self.alive = False
            return
        player = pygame.Rect(self.pos[0], self.pos[1], 100, 100)
        pipe = pygame.Rect(pipe.getPosn()[0], pipe.getPosn()[1], pipe.getDims()[0], pipe.getDims()[1])
        if player.colliderect(pipe):
            self.alive = False

    def update(self, pipe):
        self.jump -= 0.25
        self.time_alive += 1
        self.pos[1] -= self.jump
        self.died(pipe)

class Pipe:
    def __init__(self, type, height):
        self.image = ""
        self.pos = [600, height]

        self.width = 138
        self.height = 793

        if type == "up":
            self.image = pygame.image.load("testpipe_up.png")
            self.pos[1] = height - self.height
        else:
            self.image = pygame.image.load("testpipe_down.png")
            self.pos[1] = 800 - height

        self.image = pygame.transform.scale(self.image, (self.width, self.height))

        self.speed = -5
        self.onscreen = True

    def draw(self, screen):
        screen.blit(self.image, self.pos)

    def update(self):
        self.pos[0] += self.speed
        if self.pos[0] < -1 * self.width:
            self.onscreen = False

    def is_onscreen(self):
        return self.onscreen

    def getPosn(self):
        return self.pos

    def getDims(self):
        return [self.width, self.height]

def run_game(genomes, config):
    nets = []
    birds = []

    init_numb = random.randint(200, 500)
    pipes = [Pipe("down", init_numb), Pipe("up", 550 - init_numb)]

    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        birds.append(Bird())

    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 70)
    font = pygame.font.SysFont("Arial", 30)

    global generation
    generation += 1
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        clock.tick(40)
        #screen.fill((255, 255, 255))
        screen.blit(background, (0,0))

        if pipes[len(pipes) - 1].getPosn()[0] < 300 and pipes[len(pipes) - 2].getPosn()[0] < 300:
            init_numb = random.randint(200, 500)
            pipes.append(Pipe("down", init_numb))
            pipes.append(Pipe("up", 550 - init_numb))

        for index, bird in enumerate(birds):
            output = nets[index].activate(bird.get_data(pipes))
            i = output.index(max(output))
            if i == 0:
                bird.jump_up()
        living = 0
        for i, bird in enumerate(birds):
            if bird.is_alive():
                living += 1
                for pipe in pipes:
                    bird.update(pipe)
                genomes[i][1].fitness += bird.get_reward()

        if living == 0:
            break

        for bird in birds:
            if bird.is_alive():
                bird.draw(screen)

        for pipe in pipes:
            if pipe.is_onscreen():
                pipe.update()
                pipe.draw(screen)
        pipes = [pipe for pipe in pipes if pipe.is_onscreen()]
        pygame.display.flip()

if __name__ == "__main__":
    # Set configuration file
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create core evolution algorithm class
    p = neat.Population(config)

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    p.run(run_game, 1000)

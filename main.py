import pygame
import neat
import time
import os
import random

from Bird import *
from Pipe import *
from Base import *

pygame.font.init()

GEN = 0
WIN_WIDTH = 550
WIN_HEIGHT = 800

STAT_FONT = pygame.font.SysFont("comicsans", 50)

COLOR_WHITE = (255, 255, 255)


def load_image_scale2x(name):
    return pygame.transform.scale2x(pygame.image.load(os.path.join("assets", name)))


BIRD_IMAGES = [load_image_scale2x("bird1.png"), load_image_scale2x("bird2.png"), load_image_scale2x("bird2.png")]
BG_IMG = load_image_scale2x("bg.png")


def draw_window(window, birds, pipes, base, score, gen):
    window.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(window)

    score_label = STAT_FONT.render(f"Score: {str(score)}", True, COLOR_WHITE)
    window.blit(score_label, (WIN_WIDTH - 10 - score_label.get_width(), 10))

    gen_label = STAT_FONT.render(f"Generation: {str(gen)}", True, COLOR_WHITE)
    window.blit(gen_label, (10, 10))

    base.draw(window)

    for bird in birds:
        bird.draw(window)

    pygame.display.update()


def main(genomes, config):
    global GEN
    GEN += 1

    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)

    base = Base(730)
    pipes = [Pipe(700)]
    _run = True
    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    score = 0

    while _run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _run = False
                pygame.quit()
                quit()

        pipe_index = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_index = 1
        else:
            _run = False
            break

        # take the decision
        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_index].height),
                                       abs(bird.y - pipes[pipe_index].bottom)))

            if output[0] > 0.5:
                bird.jump()

        add_pipe = False
        rem = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(700))

        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        base.move()
        draw_window(window, birds, pipes, base, score, GEN)


def run(config_path):
    # set up the NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(main, 50)


# pass the configuration file to run module
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)


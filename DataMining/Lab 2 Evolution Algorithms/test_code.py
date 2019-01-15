import unittest

import cv2 as cv
import matplotlib.pyplot as plt


class TestCode(unittest.TestCase):

    def test_inspyred(self):
        import random
        import time
        import inspyred

        def generate_binary(random, args):
            bits = args.get('num_bits', 8)
            return [random.choice([0, 1]) for i in range(bits)]

        @inspyred.ec.evaluators.evaluator
        def evaluate_binary(candidate, args):
            return int("".join([str(c) for c in candidate]), 2)

        rand = random.Random()
        rand.seed(int(time.time()))
        ga = inspyred.ec.GA(rand)
        ga.observer = inspyred.ec.observers.stats_observer
        ga.terminator = inspyred.ec.terminators.evaluation_termination
        final_pop = ga.evolve(evaluator=evaluate_binary,
                              generator=generate_binary,
                              max_evaluations=1000,
                              num_elites=1,
                              pop_size=100,
                              num_bits=10)
        final_pop.sort(reverse=True)
        for ind in final_pop:
            print(str(ind))

    def test_the_generator(self):
        from random import Random
        from time import time
        from time import sleep
        import inspyred
        from tkinter import *
        import itertools

        def generate_polygon(random, args):
            size = args.get('num_vertices', 6)
            return [(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(size)]

        def segments(p):
            return zip(p, p[1:] + [p[0]])

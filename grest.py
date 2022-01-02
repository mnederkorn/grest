from game import *
# from game import printm
from pg import *
from mpg import *
from dpg import *
from eg import *
from eg_n import *
from ssg import *
from ssg_n import *
from time import time
import sys
import copy
from func_timeout import func_timeout, FunctionTimedOut
from math import log
from itertools import count

if __name__ == "__main__":

    # pg = ParityGame.generate(n, p, w)
    # mpg = MeanPayoffGame.generate(n, p, w)
    # eg = EnergyGame.generate(n, p, w)
    # dpg = DiscountedPayoffGame.generate(n, p, w)
    # ssg = SimpleStochasticGame.generate(n, p)

    n = int(8)

    p = 2 / n

    w=int(n)
    # w = int(log(n, 2))

    while True:

        mpg = MeanPayoffGame.generate(n, p, w)
        # mpg = Game.load(r"C:\ata\uni\master\grest\grest\graphs\MeanPayoffGame_2022-01-02-09-54-09.bin")
        # mpg.owner[0]=True
        # mpg.visualise()
        # mpg.save()
        x = mpg.solve_value()
        y = mpg.solve_value_eg()

        print(x,y)

        if not np.all(x==y):
            mpg.save()
            exit()
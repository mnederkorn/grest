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

    n = 4
    p = 2/n
    w = int(n**.5)

    path = r"C:\ata\uni\master\grest\grest\test.csv"

    pg = DiscountedPayoffGame.generate(n, p, w)
    print(pg.solve_value())
    print(pg.owner)


    pg.save_csv(path)

    pg=DiscountedPayoffGame.load_csv(path)
    print(pg.solve_value())
    print(pg.owner)
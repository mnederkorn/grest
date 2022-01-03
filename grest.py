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

    o, e, h = 0,0,0

    for i in count(1):

        mpg = SimpleStochasticGame.generate(n, p)

        o += np.count_nonzero(mpg.owner)
        e += np.count_nonzero(mpg.edges!=mini)
        if np.count_nonzero(mpg.owner==2)!=0:
            h += np.sum(mpg.avg)/np.count_nonzero(mpg.owner==2)

        print(o/i, e/i, h/i)

        if not np.average(np.count_nonzero(mpg.edges, 1)):
            exit()
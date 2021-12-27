from game import *
from pg import *
from mpg import *
from mpg_n import *
from dpg import *
from dpg_n import *
from eg import *
from ssg import *
import time
import sys
import copy
from func_timeout import func_timeout, FunctionTimedOut

if __name__ == '__main__':

    n = int(8)

    p = 2/n

    w=ceil(n/2)

    # pg demo
    # pg = ParityGame.generate(n, p, w)

    # mpg demo
    # mpg = MeanPayoffGame.generate(n, p, w)

    # eg demo
    # eg = EnergyGame.generate(n, p, w)

    # dpg demo
    # dpg = DiscountedPayoffGame.generate(n, p, w)

    # ssg demo
    # ssg = SimpleStochasticGame.generate(n, p)
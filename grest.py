from game import *
from pg import *
from mpg import *
from dpg import *
from eg import *
from ssg import *
import time

if __name__ == '__main__':

    n = int((2**.5)**6)

    # outdegree of every vertex has to be >=1
    # p is taken as if this wasn't the case
    # to adjust, every vertex is given at least one outgoing edge and p is rescaled afterwards
    # expected value of outdegree of vertices stays the same but the distribution changes
    # to allow for rescaling while keeping expected value of outdegrees, p has to be 1/#V<=p<=1
    p = 2/n
    p=((p*n)-1)/(n-1)

    w=10
    h=5

    # # pg demo
    # pg = ParityGame.generate(n, p, 3)
    # print(pg.solve_zielonka())

    # mpg demo
    # mpg = MeanPayoffGame.generate(n, p, w)
    # mpg.visualise()
    # print(mpg.solve_zwick_paterson())
    # mpg.to_dpg()
    # mpg.to_eg()

    # # eg demo
    # eg = EnergyGame.generate(n, p, w)
    # eg_strat_above eg.solve_strat_iter_above()
    # eg_strat_below = eg.solve_strat_iter_below()
    # eg_solve_v = eg.solve_bcdgr()
    # eg_value_v = eg.solve_value_iter()
    # print(eg_strat_above)
    # print(eg_strat_below)
    # print(eg_solve_v)
    # print(eg_value_v)

    # # dpg demo
    # dpg = DiscountedPayoffGame.generate(n, p, w)
    # dpg_strat_v = dpg.solve_strat_iter()
    # dpg_value_v = dpg.solve_value_iter()
    # ssg = dpg.to_ssg()

    # # ssg demo
    ssg = SimpleStochasticGame.generate(n, p)
    ssg_strat = ssg.solve_strat_iter()
    ssg_value = ssg.solve_value_iter()
    print(ssg_strat)
    print(ssg_value)
    print(np.max(np.abs(ssg_value-ssg_strat)))
    ssg.visualise()
    # ssg.save()

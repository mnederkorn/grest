from game import *
from pg import *
from mpg import *
from dpg import *
from eg import *
from eg_n import *
from ssg import *
from ssg_n import *
import time
import sys
import copy
from func_timeout import func_timeout, FunctionTimedOut
from math import log
from itertools import count

if __name__ == '__main__':

    n = int(2**7)

    p = 2/n

    # w=ceil(n/2)
    w = int(log(n,2))

    tt=[0,0]

    # pg = ParityGame.generate(n, p, w)
    # pg2 = ParityGame_n(pg.owner, pg.edges, pg.priority)
    # pg.solve_value_zielonka()
    # pg2.solve_value_zielonka()

    # for i in count(1):

    #     pg = ParityGame.generate(n, p, w)
    #     pg2 = ParityGame_n(pg.owner, pg.edges, pg.priority)
    #     t=time.time()
    #     pg.solve_value_zielonka()
    #     tt[0]+=time.time()-t
    #     t=time.time()
    #     pg2.solve_value_zielonka()
    #     tt[1]+=time.time()-t

    #     print([x/i for x in tt])

    # mpg = MeanPayoffGame.generate(n, p, w)
    # mpg2 = MeanPayoffGame(mpg.owner, mpg.edges)
    # mpg.solve_value_zwick_paterson_wrap()
    # mpg2.solve_value_zwick_paterson_wrap()

    # tt=[0,0]

    # for i in count(1):

    #     mpg = MeanPayoffGame.generate(n, p, w)
    #     mpg2 = MeanPayoffGame(mpg.owner, mpg.edges)
    #     t=time.time()
    #     mpg.solve_value_zwick_paterson_wrap()
    #     tt[0]+=time.time()-t
    #     t=time.time()
    #     mpg2.solve_value_zwick_paterson_wrap()
    #     tt[1]+=time.time()-t

    #     print([x/i for x in tt])

    # eg = EnergyGame.generate(n, p, w)
    # eg2 = EnergyGame_n(eg.owner, eg.edges)
    # eg.solve_both_strat_iter_below()
    # eg2.solve_both_strat_iter_below()

    # for i in count(1):

    #     eg = EnergyGame.generate(n, p, w)
    #     eg2 = EnergyGame_n(eg.owner, eg.edges)
    #     t=time.time()
    #     x=eg.solve_both_strat_iter_below()[0]
    #     tt[0]+=time.time()-t
    #     t=time.time()
    #     y=eg2.solve_both_strat_iter_below()[0]
    #     tt[1]+=time.time()-t

    #     print([x/i for x in tt])

    #     if not np.all(x==y):
    #         eg.save()
    #         eg.visualise()
    #         exit()

    # dpg = DiscountedPayoffGame.generate(n, p, w)

    ssg = SimpleStochasticGame.generate(n, p)
    ssg2 = SimpleStochasticGame_n(ssg.owner, ssg.edges, ssg.avg_chance, ssg.stopping)
    x=ssg.solve_value_kleene()
    y=ssg2.solve_value_kleene_wrap()

    for i in count(1):

        ssg = SimpleStochasticGame.generate(n, p)
        ssg2 = SimpleStochasticGame_n(ssg.owner, ssg.edges, ssg.avg_chance, ssg.stopping)
        t=time.time()
        x=ssg.solve_value_kleene()
        tt[0]+=time.time()-t
        t=time.time()
        y=ssg2.solve_value_kleene_wrap()
        tt[1]+=time.time()-t

        print([x/i for x in tt])

        if not np.all(np.max(np.abs(x-y))<1e-14):
            # ssg.save()
            # ssg.visualise()
            exit()
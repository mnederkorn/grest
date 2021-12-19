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

    n = int((2**.5)**3)

    # outdegree of every vertex has to be >=1
    # p is taken as if this wasn't the case
    # to adjust, every vertex is given at least one outgoing edge and p is rescaled afterwards
    # expected value of outdegree of vertices stays the same but the distribution changes
    # to allow for rescaling while keeping expected value of outdegrees, p has to be 1/#V<=p<=1
    p = 2/n
    p=((p*n)-1)/(n-1)

    w=10
    h=5

    # pg demo
    # pg = ParityGame.generate(n, p, 3)
    # mpg=pg.to_mpg()
    # pg.visualise()

    # mpg demo
    # mpg = MeanPayoffGame.generate(n, p, w)
    # mpg.to_dpg()
    # mpg.to_eg()

    # # eg demo
    while True:
        # eg = EnergyGame.generate(n, p, w)
        # eg = Game.load(r"C:\ata\uni\master\grest\grest\graphs\EnergyGame_2021-12-19-00-16-42.bin")
        # eg = Game.load(r"C:\ata\uni\master\grest\grest\graphs\EnergyGame_2021-12-19-01-54-53.bin")
        # eg = Game.load(r"C:\ata\uni\master\grest\grest\graphs\EnergyGame_2021-12-19-16-32-02.bin")
        eg = Game.load(r"C:\ata\uni\master\grest\grest\graphs\EnergyGame_2021-12-19-19-39-53.bin")
        # x1 = eg.solve_value_iter()
        # x2 = eg.solve_bcdgr()
        # eg.visualise()
        # eg.save()
        x3 = eg.solve_strat_iter_below()
        x4 = eg.solve_strat_iter_above()
        # print(x1)
        # print(x2)
        # print(x3)
        # print(x4)
        # exit()

        if x4=="asdf":
            eg.visualise()
            eg.save()
            exit()


        if np.any(x3!=x4):
            # print(x1)
            # print(x2)
            print(x3)
            print(x4)
            # eg.visualise()
            eg.save()
            exit()

    # # dpg demo
    # dpg = DiscountedPayoffGame.generate(n, p, w)
    # ssg = dpg.to_ssg()

    # # ssg demo
    # ssg = SimpleStochasticGame.generate(n, p)

    ######################

    # dpg = DiscountedPayoffGame.generate(n, p, w)
    # dpg_n = DiscountedPayoffGame_n(dpg.owner, dpg.edges, dpg.discount)
    # func_timeout(1000, dpg_n.solve_value_iter_matrix_wrapper)
    # func_timeout(1000, dpg_n.solve_value_iter_lin_wrapper)

    # t_t = [0,0,0,0]

    # for i in count(1):

    #     dpg = DiscountedPayoffGame.generate(n, p, w)
    #     dpg_n = DiscountedPayoffGame_n(dpg.owner, dpg.edges, dpg.discount)

    #     t=time.time()
    #     try:
    #         func_timeout(100, dpg.solve_value_iter_matrix)
    #         t_t[0]+=time.time()-t
    #     except:
    #         pass
    #     t=time.time()
    #     try:
    #         func_timeout(100, dpg.solve_value_iter_lin)
    #         t_t[1]+=time.time()-t
    #     except:
    #         pass
    #     t=time.time()
    #     try:
    #         # func_timeout(100, dpg_n.solve_value_iter_matrix, args=(dpg_n.owner,dpg_n.edges,dpg_n.discount))
    #         func_timeout(100, dpg_n.solve_value_iter_matrix_wrapper)
    #         t_t[2]+=time.time()-t
    #     except:
    #         pass
    #     t=time.time()
    #     try:
    #         # func_timeout(100, dpg_n.solve_value_iter_lin, args=(dpg_n.owner,dpg_n.edges,dpg_n.discount))
    #         func_timeout(100, dpg_n.solve_value_iter_lin_wrapper)
    #         t_t[3]+=time.time()-t
    #     except:
    #         pass

    #     print([f"{n/i:.5f}" for n in t_t], f"{t_t[0]/t_t[1]:.2f},{t_t[2]/t_t[3]:.2f}")

    ######################

    # mpg = MeanPayoffGame.generate(n, p, w)
    # mpg_n = MeanPayoffGame_n(mpg.owner, mpg.edges)

    # func_timeout(10, mpg.solve_value_iter_matrix)
    # func_timeout(10, mpg.solve_value_iter_lin)
    # func_timeout(1000, mpg_n.solve_value_iter_matrix_wrapper)
    # func_timeout(1000, mpg_n.solve_value_iter_lin_wrapper)

    # t_t = [0,0,0,0]

    # for i in count(1):


    #     mpg = MeanPayoffGame.generate(n, p, w)
    #     mpg_n = MeanPayoffGame_n(mpg.owner, mpg.edges)

    #     t=time.time()
    #     try:
    #         func_timeout(100, mpg.solve_value_iter_matrix)
    #         t_t[0]+=time.time()-t
    #     except:
    #         pass
    #     t=time.time()
    #     try:
    #         func_timeout(100, mpg.solve_value_iter_lin)
    #         t_t[1]+=time.time()-t
    #     except:
    #         pass
    #     t=time.time()
    #     try:
    #         func_timeout(100, mpg_n.solve_value_iter_matrix_wrapper)
    #         t_t[2]+=time.time()-t
    #     except:
    #         pass
    #     t=time.time()
    #     try:
    #         func_timeout(100, mpg_n.solve_value_iter_lin_wrapper)
    #         t_t[3]+=time.time()-t
    #     except:
    #         pass

    #     print([f"{n/i:.5f}" for n in t_t], f"{t_t[0]/t_t[1]:.2f},{t_t[2]/t_t[3]:.2f}")
    #     # print([f"{n/i:.5f}" for n in t_t])

    ######################

    # eg = EnergyGame.generate(n, p, w)
    # eg_n = EnergyGame_n(eg.owner, eg.edges)

    # func_timeout(10, eg.solve_bcdgr)
    # func_timeout(1000, eg_n.solve_value_iter_wrapper)

    # t_t = [0,0,0,0]

    # for i in count(1):

    #     eg = EnergyGame.generate(n, p, w)
    #     eg_n = EnergyGame_n(eg.owner, eg.edges)

    #     t=time.time()
    #     try:
    #         func_timeout(1, eg.solve_bcdgr)
    #         t_t[0]+=time.time()-t
    #     except:
    #         pass
    #     t=time.time()
    #     try:
    #         func_timeout(10, eg_n.solve_value_iter_wrapper)
    #         t_t[1]+=time.time()-t
    #     except:
    #         pass

    #     # print([f"{n/i:.5f}" for n in t_t], f"{t_t[2]/t_t[3]:.2f}")
    #     print([f"{n/i:.5f}" for n in t_t])
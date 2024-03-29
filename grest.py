from game import *
from pg import *
from mpg import *
from dpg import *
from eg import *
from ssg import *
from math import log
from os import remove

np.set_printoptions(linewidth=100000)

if __name__ == "__main__":

    n = 6
    p = log(n, 2) / n
    w = n

    print("pg " + "#" * 100)
    pg = ParityGame.generate(n, p, w)
    pg_visualise = pg.visualise()
    pg_save_csv = pg.save_csv()
    pg_save_bin = pg.save_bin()
    pg = pg.load_csv(pg_save_csv)
    pg = pg.load_bin(pg_save_bin)
    remove(pg_visualise)
    remove(pg_save_csv)
    remove(pg_save_bin)
    print("strat zielonka    ", pg_strat_1 := pg._solve_both_zielonka()[1])
    print("strat via mpg     ", pg_strat_2 := pg.solve_strat_mpg())
    print("-" * 100)
    print("value zielonka              ", pg._solve_both_zielonka()[0])
    print("value via mpg               ", pg.solve_value_mpg())
    print("-" * 100)
    print("value via strat zielonka    ", pg.solve_value(pg_strat_1))
    print("value via strat via mpg     ", pg.solve_value(pg_strat_2))
    print("-" * 100)

    print("mpg " + "#" * 100)
    mpg = MeanPayoffGame.generate(n, p, w)
    mpg_visualise = mpg.visualise()
    mpg_save_csv = mpg.save_csv()
    mpg_save_bin = mpg.save_bin()
    mpg = mpg.load_csv(mpg_save_csv)
    mpg = mpg.load_bin(mpg_save_bin)
    remove(mpg_visualise)
    remove(mpg_save_csv)
    remove(mpg_save_bin)
    print("strat zp         ", mpg_strat_1 := mpg._solve_strat_zwick_paterson())
    print("strat via eg     ", mpg_strat_2 := mpg.solve_both_eg()[1])
    print("strat via dpg    ", mpg_strat_3 := mpg.solve_both_dpg()[1])
    print("-" * 100)
    print("value zp                   ", mpg._solve_value_zwick_paterson())
    print("value via eg               ", mpg.solve_both_eg()[0])
    print("value via dpg              ", mpg.solve_both_dpg()[0])
    print("value via strat zp         ", mpg.solve_value(mpg_strat_1))
    print("value via strat via eg     ", mpg.solve_value(mpg_strat_2))
    print("value via strat via dpg    ", mpg.solve_value(mpg_strat_3))
    print("#" * 100)

    print("eg")
    eg = EnergyGame.generate(n, p, w)
    eg_visualise = eg.visualise()
    eg_save_csv = eg.save_csv()
    eg_save_bin = eg.save_bin()
    eg = eg.load_csv(eg_save_csv)
    eg = eg.load_bin(eg_save_bin)
    remove(eg_visualise)
    remove(eg_save_csv)
    remove(eg_save_bin)
    print("strat bcdgr               ", eg_strat_1 := eg._solve_both_bcdgr()[1])
    print("strat fpi                 ", eg_strat_2 := eg._solve_strat_fpi())
    print("strat strat iter above    ", eg_strat_3 := eg._solve_both_strat_iter_above()[1])
    print("strat strat iter below    ", eg_strat_4 := eg._solve_both_strat_iter_below()[1])
    print("-" * 100)
    print("value bcdgr                         ", eg._solve_both_bcdgr()[0])
    print("value fpi                           ", eg._solve_value_fpi())
    print("value strat iter above              ", eg._solve_both_strat_iter_above()[0])
    print("value strat iter below              ", eg._solve_both_strat_iter_below()[0])
    print("value via strat bcdgr               ", eg.solve_value(eg_strat_1))
    print("value via strat fpi                 ", eg.solve_value(eg_strat_2))
    print("value via strat strat iter above    ", eg.solve_value(eg_strat_3))
    print("value via strat strat iter below    ", eg.solve_value(eg_strat_4))
    print("#" * 100)

    print("dpg")
    dpg = DiscountedPayoffGame.generate(n, p, w)
    dpg_visualise = dpg.visualise()
    dpg_save_csv = dpg.save_csv()
    dpg_save_bin = dpg.save_bin()
    dpg = dpg.load_csv(dpg_save_csv)
    dpg = dpg.load_bin(dpg_save_bin)
    remove(dpg_visualise)
    remove(dpg_save_csv)
    remove(dpg_save_bin)
    print("strat fpi                           ", dpg_strat_1 := dpg._solve_both_fpi()[1])
    print("strat strat iter above              ", dpg_strat_2 := dpg._solve_both_strat_iter(0)[1])
    print("strat strat iter below              ", dpg_strat_3 := dpg._solve_both_strat_iter(1)[1])
    print("strat via ssg                       ", dpg_strat_4 := dpg.solve_both_ssg()[1])
    print("-" * 100)
    print("value fpi                           ", dpg._solve_both_fpi()[0])
    print("value strat iter above              ", dpg._solve_both_strat_iter(0)[0])
    print("value strat iter below              ", dpg._solve_both_strat_iter(1)[0])
    print("value via ssg                       ", dpg.solve_both_ssg()[0])
    print("value via strat fpi                 ", dpg.solve_value(dpg_strat_1))
    print("value via strat strat iter above    ", dpg.solve_value(dpg_strat_2))
    print("value via strat strat iter below    ", dpg.solve_value(dpg_strat_3))
    print("value via strat via ssg             ", dpg.solve_value(dpg_strat_4))
    print("#" * 100)

    print("ssg")
    ssg = DiscountedPayoffGame.generate(int(n / (2 ** 0.5)), p, w)._to_ssg()[0]
    # ssg = SimpleStochasticGame.generate(n, p)
    ssg_visualise = ssg.visualise()
    ssg_save_csv = ssg.save_csv()
    ssg_save_bin = ssg.save_bin()
    ssg = ssg.load_csv(ssg_save_csv)
    ssg = ssg.load_bin(ssg_save_bin)
    remove(ssg_visualise)
    remove(ssg_save_csv)
    remove(ssg_save_bin)
    print("strat fpi                           ", ssg_strat_1 := ssg._solve_both_fpi()[1])
    print("strat strat iter above              ", ssg_strat_2 := ssg._solve_both_strat_iter(0)[1])
    print("strat strat iter below              ", ssg_strat_3 := ssg._solve_both_strat_iter(1)[1])
    print("-" * 100)
    print("value fpi                           ", ssg._solve_both_fpi()[0])
    print("value strat iter above              ", ssg._solve_both_strat_iter(0)[0])
    print("value strat iter below              ", ssg._solve_both_strat_iter(1)[0])
    print("value via strat fpi                 ", ssg.solve_value(ssg_strat_1))
    print("value via strat strat iter above    ", ssg.solve_value(ssg_strat_2))
    print("value via strat strat iter below    ", ssg.solve_value(ssg_strat_3))
    print("#" * 100)

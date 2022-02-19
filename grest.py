from game import *
from pg import *
from mpg import *
from dpg import *
from eg import *
from ssg import *
from math import log

n = 16
p = log(n, 2) / n
w = n

# print("pg")
# pg = ParityGame.generate(n, p, w)
# print(pg.solve_value_zielonka())
# print(pg.solve_strat_zielonka())
# print(pg.solve_both_mpg(0))
# print(pg.solve_strat_mpg())
# print(x := pg.solve_strat())
# print(pg.solve_value())
# print(pg.solve_value(x))

# print("mpg")
# mpg = MeanPayoffGame.generate(n, p, w)
# print(mpg.solve_value_zwick_paterson_wrap())
# print(mpg.solve_strat_zwick_paterson())
# print(mpg.solve_both_dpg(0))
# print(mpg.solve_both_eg())
# print(x := mpg.solve_strat())
# print(mpg.solve_value())
# print(mpg.solve_value(x))

# print("eg")
# eg = EnergyGame.generate(n, p, w)
# print(eg.solve_both_bcdgr_wrap())
# print(eg.solve_value_kleene())
# print(eg.solve_strat_kleene())
# print(eg.solve_both_strat_iter_below())
# print(eg.solve_both_strat_iter_above())
# print(x := eg.solve_strat())
# print(eg.solve_value())
# print(eg.solve_value(x))

# print("dpg")
# dpg = DiscountedPayoffGame.generate(n, p, w)
# print(dpg.solve_both_ssg())
# print(dpg.solve_both_kleene_wrap())
# print(dpg.solve_both_strat_iter(0))
# print(x := dpg.solve_strat())
# print(dpg.solve_value())
# print(dpg.solve_value(x))

# print("ssg")
# ssg = SimpleStochasticGame.generate(n, p)
# print(ssg.solve_both_kleene_wrap())
# print(DiscountedPayoffGame.generate(n, p, w).to_ssg()[0].solve_both_strat_iter(0))
# print(x := ssg.solve_strat())
# print(ssg.solve_value())
# print(ssg.solve_value(x))


while True:

    eg = EnergyGame.generate(n, p, w)

    x = eg.solve_both_bcdgr_wrap()[1]
    y = eg.solve_both_strat_iter_above()[1]

    print(eg.solve_value())
    x = eg.solve_value(x)
    y = eg.solve_value(y)

    if not np.all(x == y):
        exit()

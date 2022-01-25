from game import *
from pg import *
from mpg import *
from dpg import *
from eg import *
from ssg import *
from time import time
import sys
import copy
from func_timeout import func_timeout, FunctionTimedOut
from math import log
from itertools import count
import winsound
import math

n = 8
p = 2/n
w = n

fp = r"C:\ata\uni\master\grest\grest\graphs"

# pg = ParityGame.generate(n, p, w)
# print(pg.solve_value_zielonka())
# print(pg.solve_strat_zielonka())
# print(pg.solve_both_mpg(0))
# print(pg.solve_strat_mpg())
# print(x:=pg.solve_strat())
# print(pg.solve_value())
# print(pg.solve_value(x))

# mpg = MeanPayoffGame.generate(n, p, w)
# print(mpg.solve_value_zwick_paterson_wrap())
# print(mpg.solve_strat_zwick_paterson())
# print(mpg.solve_both_dpg(0))
# print(mpg.solve_both_eg())
# print(x:=mpg.solve_strat())
# print(mpg.solve_value())
# print(mpg.solve_value(x))

# eg = EnergyGame.generate(n, p, w)
# print(eg.solve_both_bcdgr_wrap())
# print(eg.solve_value_kleene())
# print(eg.solve_strat_kleene())
# print(eg.solve_both_strat_iter_below())
# print(eg.solve_both_strat_iter_above())
# print(x:=eg.solve_strat())
# print(eg.solve_value())
# print(eg.solve_value(x))

# # infeasible example
# # dpg = Game.load(os.path.join(fp,r"DiscountedPayoffGame_2022-01-24-23-33-37.bin"))
# dpg = DiscountedPayoffGame.generate(n, p, w)
# print(dpg.solve_both_ssg())
# print(dpg.solve_both_kleene_wrap())
# print(dpg.solve_both_strat_iter(0))
# print(x:=dpg.solve_strat())
# print(dpg.solve_value())
# print(dpg.solve_value(x))

# ssg = SimpleStochasticGame.generate(n, p)
# print(ssg.solve_both_kleene_wrap())
# print(DiscountedPayoffGame.generate(n, p, w).to_ssg()[0].solve_both_strat_iter(0))
# print(x:=ssg.solve_strat())
# print(ssg.solve_value())
# print(ssg.solve_value(x))




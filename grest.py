from game import *
from pg import *
from mpg import *
from dpg import *
from eg import *
from ssg import *
from math import log
from os import remove

n = 8
p = log(n, 2) / n
w = n

print("pg")
pg = ParityGame.generate(n, p, w)
pg_save = pg.save_csv()
pg = pg.load_csv(pg_save)
remove(pg_save)
print(pg_strat_1 := pg.solve_strat_zielonka())
print(pg_strat_2 := pg.solve_strat_mpg())
print(pg.solve_value_zielonka())
print(pg.solve_value_mpg())
print(pg.solve_value(pg_strat_1))
print(pg.solve_value(pg_strat_2))

print("mpg")
mpg = MeanPayoffGame.generate(n, p, w)
mpg_save = mpg.save_csv()
mpg = mpg.load_csv(mpg_save)
remove(mpg_save)
print(mpg_strat_1 := mpg.solve_strat_zwick_paterson())
print(mpg_strat_2 := mpg.solve_both_dpg()[1])
print(mpg_strat_3 := mpg.solve_both_eg()[1])
print(mpg.solve_value_zwick_paterson_wrap())
print(mpg.solve_both_dpg()[0])
print(mpg.solve_both_eg()[0])
print(mpg.solve_value(mpg_strat_1))
print(mpg.solve_value(mpg_strat_2))
print(mpg.solve_value(mpg_strat_3))

print("eg")
eg = EnergyGame.generate(n, p, w)
eg_save = eg.save_csv()
eg = eg.load_csv(eg_save)
remove(eg_save)
print(eg_strat_1 := eg.solve_both_bcdgr_wrap()[1])
print(eg_strat_2 := eg.solve_strat_kleene())
print(eg_strat_3 := eg.solve_both_strat_iter_below()[1])
print(eg_strat_4 := eg.solve_both_strat_iter_above()[1])
print(eg.solve_both_bcdgr_wrap()[0])
print(eg.solve_value_kleene())
print(eg.solve_both_strat_iter_below()[0])
print(eg.solve_both_strat_iter_above()[0])
print(eg.solve_value(eg_strat_1))
print(eg.solve_value(eg_strat_2))
print(eg.solve_value(eg_strat_3))
print(eg.solve_value(eg_strat_4))

print("dpg")
dpg = DiscountedPayoffGame.generate(n, p, w)
dpg_save = dpg.save_csv()
dpg = dpg.load_csv(dpg_save)
remove(dpg_save)
print(dpg_strat_1 := dpg.solve_both_kleene_wrap()[1])
print(dpg_strat_2 := dpg.solve_both_strat_iter(0)[1])
print(dpg_strat_3 := dpg.solve_both_strat_iter(1)[1])
print(dpg.solve_both_kleene_wrap()[0])
print(dpg.solve_both_strat_iter(0)[0])
print(dpg.solve_both_strat_iter(1)[0])
print(dpg.solve_value(dpg_strat_1))
print(dpg.solve_value(dpg_strat_2))
print(dpg.solve_value(dpg_strat_3))

print("ssg")
ssg = DiscountedPayoffGame.generate(int(n / (2 ** 0.5)), p, w).to_ssg()[0]
ssg_save = ssg.save_csv()
ssg = ssg.load_csv(ssg_save)
remove(ssg_save)
print(ssg_strat_1 := ssg.solve_both_kleene_wrap()[1])
print(ssg_strat_2 := ssg.solve_both_strat_iter(0)[1])
print(ssg_strat_3 := ssg.solve_both_strat_iter(1)[1])
print(ssg.solve_both_kleene_wrap()[0])
print(ssg.solve_both_strat_iter(0)[0])
print(ssg.solve_both_strat_iter(1)[0])
print(ssg.solve_value(ssg_strat_1))
print(ssg.solve_value(ssg_strat_2))
print(ssg.solve_value(ssg_strat_3))

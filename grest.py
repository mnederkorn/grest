from game import *
# from game import printm
from pg import *
from mpg import *
from dpg import *
from eg import *
from ssg import *
from ssg_n import *
from time import time
import sys
import copy
from func_timeout import func_timeout, FunctionTimedOut
from math import log
from itertools import count
import winsound
import math

# pg = ParityGame.generate(n, p, w)
# mpg = MeanPayoffGame.generate(n, p, w)
# eg = EnergyGame.generate(n, p, w)
# dpg = DiscountedPayoffGame.generate(n, p, w)
# ssg = SimpleStochasticGame.generate(n, p)

TIMES = 100
n = [2**i for i in range(2, 7)]
p = [lambda x: 1/x, lambda x: log(x,2)/x, lambda x: .5]
pd = {p[0]:"1/x",p[1]:"log2/x",p[2]:".5"}
w = [lambda x: int(log(x,2)),lambda x: int(x),lambda x: int(int(log(x,2))*x)]
wd = {w[0]:"log2",w[1]:"x",w[2]:"x*log2"}

path = r"C:\ata\uni\master\grest\grest\tests"

g = EnergyGame.generate(8, .5, 8)
g.solve_strat_kleene()

with open(os.path.join(path,"eg_strat_kleene_100.dat"), "w") as file:
    for a in n:
        for b in p:
            for c in w:
                tt=0
                for n in range(TIMES):
                    print(a, pd[b], n,tt)
                    g = EnergyGame.generate(a, b(a), c(a))
                    t=time()
                    x=g.solve_strat_kleene()
                    tt+=time()-t
                file.write(",".join((str(x) for x in (a, pd[b], wd[c], tt)))+"\n")
                file.flush()

for _ in range(10):
    winsound.Beep(220, 50)
from game import *
import numpy as np
from math import floor,ceil
from dpg import DiscountedPayoffGame
from eg import EnergyGame
from graphviz import Digraph
from tempfile import gettempdir
from itertools import count
import time
import copy
from numba import jit

@jit(nopython=True, cache=True)
def numba_min_axis1(x):
    out = np.empty(x.shape[0], dtype=x.dtype)
    for i in range(x.shape[0]):
        out[i] = np.min(x[i])
    return out

@jit(nopython=True, cache=True)
def numba_nanmin_axis1(x):
    out = np.empty(x.shape[0], dtype=x.dtype)
    for i in range(x.shape[0]):
        out[i] = np.nanmin(x[i])
    return out

@jit(nopython=True, cache=True)
def numba_max_axis1(x):
    out = np.empty(x.shape[0], dtype=x.dtype)
    for i in range(x.shape[0]):
        out[i] = np.max(x[i])
    return out

@jit(nopython=True, cache=True)
def numba_nanmax_axis1(x):
    out = np.empty(x.shape[0], dtype=x.dtype)
    for i in range(x.shape[0]):
        out[i] = np.nanmax(x[i])
    return out    @staticmethod

@jit(nopython=True, cache=True)
def trunc(n, v):

    lower = v-(1/(2*n*(n-1)))
    upper = v+(1/(2*n*(n-1)))

    v = np.full(n, 0, dtype=np.float64)

    for v_n in range(n):
        for denominator in range(1,n+1):
            f=floor(lower[v_n]*denominator)
            c=ceil(upper[v_n]*denominator)
            num = np.arange(f,c+1)/denominator
            if np.any((lower[v_n]<num)&(upper[v_n]>num)):
                v[v_n] = num[((lower[v_n]<num)&(upper[v_n]>num))][0]
                break

    return v

def Q_sub_sup(n, mid):

    r = np.arange(1,n+1)

    l = r*mid

    l1 = np.floor(l).astype(int)
    l2 = np.ceil(l).astype(int)

    ar1 = np.argmax(l1/r)
    ar2 = np.argmin(l2/r)

    return np.array([l1[ar1], r[ar1]], dtype=int),np.array([l2[ar2], r[ar2]], dtype=int)

def solve_value_eg_alg2(owner, edges, lower, upper):

    mini = np.iinfo(edges.dtype).min

    a1, a2 = Q_sub_sup(len(owner), (lower+upper)/2)

    e1 = np.where(edges != mini, (a1[1]*edges)-a1[0], mini)
    e2 = np.where(edges != mini, (-a1[1]*edges)+a1[0], mini)
    e3 = np.where(edges != mini, (a2[1]*edges)-a2[0], mini)
    e4 = np.where(edges != mini, (-a2[1]*edges)+a2[0], mini)
    
    f1, s1 = EnergyGame(owner, e1).solve_both()
    f2, s2 = EnergyGame(~owner, e2).solve_both()
    f3, s3 = EnergyGame(owner, e3).solve_both()
    f4, s4 = EnergyGame(~owner, e4).solve_both()

    v = np.empty(len(owner), dtype=float)
    s = np.full(len(owner), -1, dtype=int)

    v = np.where((f1!=-1)&(f2!=-1), a1[0]/a1[1], v)
    v = np.where((f3!=-1)&(f4!=-1), a2[0]/a2[1], v)
    s = np.where((f1!=-1)&(f2!=-1), np.where(s1!=-1, s1, s2), s)
    s = np.where((f3!=-1)&(f4!=-1), np.where(s1!=-1, s3, s4), s)

    v1 = np.where(f1==-1)[0]
    v2 = np.where(f4==-1)[0]

    if len(v1)!=0:
        v[v1], st = solve_value_eg_alg2(owner[v1], edges[np.ix_(v1,v1)], lower, a1[0]/a1[1])
        s[v1] = v1[st]
    if len(v2)!=0:
        v[v2], st = solve_value_eg_alg2(owner[v2], edges[np.ix_(v2,v2)], a2[0]/a2[1], upper)
        s[v2] = v2[st]

    return v, s

class MeanPayoffGame(Game):

    def __init__(self, owner, edges):

        super().__init__(owner, edges)

    @classmethod
    def generate(cls, n, p, w):

        return cls(*super().generate(n, p, w))

    def solve_value_zwick_paterson_wrap(self):

        return self.solve_value_zwick_paterson(self.owner, self.edges)

    @staticmethod
    @jit(nopython=True, cache=True)
    def solve_value_zwick_paterson(owner, edges):

        mini = np.iinfo(edges.dtype).min

        W = np.max(np.abs(np.where(edges!=mini, edges, 0)))

        v = np.zeros(len(owner), dtype=np.int64)

        k = (4*(len(owner)**3)*W)

        edges = edges.astype(np.int64)

        mini2 = np.iinfo(edges.dtype).min
        maxi2 = np.iinfo(edges.dtype).max

        # this can collide at around 4*(n^3)*(W^2)>2*62 because maxi2-((k*W)+1) can be smaller than v_k and the valid/invalid checks for edges stop making sense; analgous for mini2
        # for W = log_2(|V|) that's around |V|=160000
        # for W = |V|^(1/2) that's around |V|=32768=2^15
        # for W = |V| that's around |V|=4096=2^12

        x=np.where(edges!=mini, edges, maxi2-((k*W)+1))
        y=np.where(edges!=mini, edges, mini2+((k*W)+1))

        # edges = np.where(owner, x, y)

        edges = np.empty((len(owner),len(owner)), dtype=edges.dtype)
        for n,(i,j,l) in enumerate(zip(owner, x, y)):
            if i:
                edges[n]=j
            else:
                edges[n]=l

        for _ in np.arange(k):

            edges_weight = edges+v

            v = np.where(owner, numba_min_axis1(edges_weight), numba_max_axis1(edges_weight))

        v = v/k

        return trunc(len(owner), v)

    def solve_strat_zwick_paterson(self):

        mini = np.iinfo(self.edges.dtype).min

        z = self.solve_value_zwick_paterson_wrap()

        ret = np.full(len(self.owner), -1, dtype=int)
        edges = np.array(self.edges)

        for i,v in enumerate(edges):
            w = np.where(v!=mini)[0]
            while True:
                cl = ceil(len(w)/2)
                one,two = w[:cl],w[cl:]
                e = edges.copy()
                e[i]=mini
                e[i,one]=edges[i,one]
                x = MeanPayoffGame(self.owner, e).solve_value_zwick_paterson_wrap()
                if np.all(x==z):
                    if len(one)==1:
                        ret[i]=one[0]
                        break
                    else:
                        w = one
                else:
                    w = two
            tmp = edges[i,ret[i]]
            edges[i]=mini
            edges[i,ret[i]]=tmp
        return ret

    def to_dpg(self):

        mini = np.iinfo(self.edges.dtype).min

        W = np.max(np.abs(np.where(self.edges != mini, self.edges, 0)))

        discount = 1-(1/(4*(len(self.owner)**3)*W))

        return DiscountedPayoffGame(self.owner, self.edges, discount)

    def solve_value_dpg(self):

        dpg = self.to_dpg()

        v = dpg.solve_value()

        return trunc(len(self.owner), v)

    def solve_strat_dpg(self):

        dpg = self.to_dpg()

        return dpg.solve_strat()

    def solve_both_eg(self):

        mini = np.iinfo(self.edges.dtype).min

        W = np.max(np.abs(np.where(self.edges != mini, self.edges, 0)))

        return solve_value_eg_alg2(self.owner, self.edges, -W, W)

    def solve_value(self, strat=None):

        if type(strat) != type(None):

            mini = np.iinfo(self.edges.dtype).min

            edges = np.where((strat==-1).reshape(-1,1), self.edges, mini)

            for i in np.where(strat!=-1)[0]:
                edges[i,strat[i]]=self.edges[i,strat[i]]

            return MeanPayoffGame(self.owner, edges).solve_value_zwick_paterson_wrap()

        else:

            return self.solve_value_zwick_paterson_wrap()

    def solve_strat(self):

        return self.solve_strat_zwick_paterson()

    def visualise(self, target_path=None, strat=None, values=None):

        if type(strat) == type(None):
            strat = np.full(len(self.owner),-1)

        if target_path == None:
            target_path = os.path.join(gettempdir(), f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")

        view = Digraph(format="png")
        view.attr(bgcolor="#f0f0f0")
        for i,owner in enumerate(self.owner):
            if type(values) != type(None):
                label = f"<v(v<sub>{i}</sub>)={float(values[i]):.2f}>"
            else:
                label = f"<v<sub>{i}</sub>>"
            view.node(f"{i}", label=label, shape=shape[owner], fontcolor=colour[owner][strat[i]!=-1], color=colour[owner][strat[i]!=-1])
        mini = np.iinfo(self.edges.dtype).min
        mini = np.iinfo(self.edges.dtype).min
        idx = np.where(self.edges!=mini)
        for s,t in zip(idx[0],idx[1]):
            view.edge(str(s),str(t),str(self.edges[s,t]), fontcolor=colour[self.owner[s]][strat[s]==t], color=colour[self.owner[s]][strat[s]==t])
                    
        save_loc = view.render(filename=target_path, view=False, cleanup=True)

        return save_loc

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

class MeanPayoffGame(Game):

    def __init__(self, owner, edges):

        super().__init__(owner, edges)
        
    @classmethod
    def generate(cls, n, p, w):
        assert p>=1/n, "Since |post(v)| needs to be >=1 for every v, p needs to be at least p>=1/n"
        p=((p*n)-1)/(n-1)
        owner = np.random.choice([False, True], size=(n))
        edges_exist = np.empty((n,n), dtype=bool)
        for e in edges_exist:
            rng = np.random.randint(n)
            e[rng] = True
            e[:rng] = np.random.choice([False, True], size=rng, p=[1-p, p])
            e[rng+1:] = np.random.choice([False, True], size=n-(rng+1), p=[1-p, p])
        edges_value = np.random.randint(-w, w+1, size=(n,n))
        mini = np.iinfo(edges_value.dtype).min
        edges = np.where(edges_exist, edges_value, mini)

        return cls(owner, edges)

    def _solve_value_kleene(self):

        mini = np.iinfo(self.edges.dtype).min

        cur = np.zeros(len(self.owner)).reshape((-1,1))

        for i in count(1):

            old = np.array(cur)

            edges_weight = np.where(self.edges != mini, (self.edges/i)+cur*((i-1)/i), np.nan)

            cur = np.where(self.owner, np.nanmin(edges_weight, 1), np.nanmax(edges_weight, 1))

            max_err = np.amax(np.abs(cur-old))

            # iterate until max float precision is hit
            if max_err <1e-4:
                break

        # return cur, strat
        return cur

    def solve_value_zwick_paterson(self):

        mini = np.iinfo(self.edges.dtype).min

        W = np.max(np.abs(np.where(self.edges!=mini, self.edges, 0)))

        v = np.zeros(len(self.owner), dtype=np.int64)

        k = (4*(len(self.owner)**3)*W)

        edges = self.edges.astype(np.int64)

        mini2 = np.iinfo(edges.dtype).min
        maxi2 = np.iinfo(edges.dtype).max

        # this can collide at around 4*(n^3)*(W^2)>2*62 because maxi2-((k*W)+1) can be smaller than v_k and the valid/invalid checks for edges stop making sense; analgous for mini2
        # for W = log_2(|V|) that's around |V|=160000
        # for W = |V|^(1/2) that's around |V|=32768=2^15
        # for W = |V| that's around |V|=4096=2^12

        edges = np.where(self.owner.reshape(-1,1), np.where(edges!=mini, edges, maxi2-((k*W)+1)), np.where(edges!=mini, edges, mini2+((k*W)+1)))

        for x in range(k):

            edges_weight = edges+v

            v = np.where(self.owner, np.min(edges_weight, 1), np.max(edges_weight, 1))

        v = v/k

        lower = v-(1/(2*len(self.owner)*(len(self.owner)-1)))
        upper = v+(1/(2*len(self.owner)*(len(self.owner)-1)))

        v = np.zeros(len(self.owner), dtype=float)

        for v_n in range(len(self.owner)):
            for denominator in range(1,len(self.owner)+1):
                f=floor(lower[v_n]*denominator)
                c=ceil(upper[v_n]*denominator)
                num = np.arange(f,c+1)/denominator
                if np.any((lower[v_n]<num)&(upper[v_n]>num)):
                    v[v_n] = num[((lower[v_n]<num)&(upper[v_n]>num))][0]
                    break

        return v

    def solve_strat_zwick_paterson(self):

        mini = np.iinfo(self.edges.dtype).min

        z = self.solve_value_zwick_paterson()

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
                x = MeanPayoffGame(self.owner, e).solve_value_zwick_paterson()
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

    def solve_value(self, strat=None):

        if type(strat) != type(None):

            mini = np.iinfo(self.edges.dtype).min

            edges = np.where((strat==-1).reshape(-1,1), self.edges, mini)

            for i in np.where(strat!=-1)[0]:
                edges[i,strat[i]]=self.edges[i,strat[i]]

            return MeanPayoffGame(self.owner, edges).solve_value_zwick_paterson()

        else:

            return self.solve_value_zwick_paterson()

    def solve_strat(self):

        return self.solve_strat_zwick_paterson()

    def to_dpg(self):

        mini = np.iinfo(self.edges.dtype).min

        empty_as_zero = np.where(self.edges != mini, self.edges, 0)

        W = max(abs(np.amin(empty_as_zero)),abs(np.amax(empty_as_zero)))

        discount = 1-(1/(4*(len(self.owner)**3)*W))

        return DiscountedPayoffGame(self.owner, self.edges, np.array((discount,)))

    def to_eg(self):

        return EnergyGame(self.owner, self.edges)

    def visualise(self, target_path=None, strat=None, values=None):

        if type(strat) == type(None):
            strat = np.full(len(self.owner),-1)

        if target_path == None:
            target_path = os.path.join(gettempdir(), f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")

        view = Digraph(format="png")
        view.attr(bgcolor="#f0f0f0")
        for i,owner in enumerate(self.owner):
            label = f"<v(v<sub>{i}</sub>)={float(values[i]):.2f}>"
            view.node(f"{i}", label=label, shape=shape[owner], fontcolor=colour[owner][strat[i]!=-1], color=colour[owner][strat[i]!=-1])
        mini = np.iinfo(self.edges.dtype).min
        mini = np.iinfo(self.edges.dtype).min
        idx = np.where(self.edges!=mini)
        for s,t in zip(idx[0],idx[1]):
            view.edge(str(s),str(t),str(self.edges[s,t]), fontcolor=colour[self.owner[s]][strat[s]==t], color=colour[self.owner[s]][strat[s]==t])
                    
        save_loc = view.render(filename=target_path, view=False, cleanup=True)

        return save_loc

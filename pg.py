from game import *
import numpy as np
from mpg import MeanPayoffGame
from tempfile import gettempdir
from graphviz import Digraph
from math import ceil
from itertools import count
from time import sleep
from numba import jit

e_o = {0:"Even",1:"Odd"}

@jit(nopython=True, cache=True)
def numba_any_axis1(x):
    """Numba compatible version of np.any(x, axis=1)."""
    out = np.zeros(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_or(out, x[:, i])
    return out

@jit(nopython=True, cache=True)
def find_attractor(player, owner, edges, Nh):

    if len(Nh)==0:
        return np.array([False])[:0]

    us = (owner==player)
    them = (owner!=player)
    
    atr = Nh.copy()

    while True:
        old = atr.copy()
        us_add = numba_any_axis1(edges[us][:,atr])

        them_add = ~numba_any_axis1(edges[them][:,~atr])
        atr[(np.where(us)[0])[us_add]] = True
        atr[(np.where(them)[0])[them_add]] = True
        if np.all(atr==old):
            break
    return atr

def zielonka(owner, edges, priority):

    if len(owner) == 0:
        return np.array([], dtype=bool)
    else:
        m = np.max(priority)
        player = m%2
        A = find_attractor(player, owner, edges, (priority==m))
        z1 = zielonka(owner[~A], edges[np.ix_(~A,~A)], priority[~A])
        if (player and (~np.any(~z1))) or (not player and (~np.any(z1))):
            return np.ones(len(owner), dtype=bool) if player else np.zeros(len(owner), dtype=bool)
        else:
            x = np.zeros(len(owner), dtype=bool)
            y = np.zeros(len(owner), dtype=bool)
            x[(np.where(~A)[0])[~z1]] = True
            y[(np.where(~A)[0])[z1]] = True
            B = find_attractor(1-player, owner, edges, x if player else y)
            z2 = zielonka(owner[~B], edges[np.ix_(~B,~B)], priority[~B])
            if player:
                a = np.zeros(len(owner), dtype=bool)
                a[(np.where(~B)[0])[z2]] = True
                return a
            else:
                b = np.ones(len(owner), dtype=bool)
                b[(np.where(~B)[0])[~z2]] = False
                return b

class ParityGame(Game):

    def __init__(self, owner, edges, priority):

        super().__init__(owner, edges)
        self.priority = priority

    @classmethod
    def generate(cls, n, p, h):
        assert p>=1/n, "Since |post(v)| needs to be >=1 for every v, p needs to be at least p>=1/n"
        p=((p*n)-1)/(n-1)
        owner = np.random.choice([False, True], size=(n))
        edges = np.empty((n,n), dtype=bool)
        for e in edges:
            rng = np.random.randint(n)
            e[rng] = True
            e[:rng] = np.random.choice([False, True], size=rng, p=[1-p, p])
            e[rng+1:] = np.random.choice([False, True], size=n-(rng+1), p=[1-p, p])
        priority = np.random.randint(0, h, size=(n))

        return cls(owner, edges, priority)

    def solve_value_zielonka(self):

        return zielonka(self.owner, self.edges, self.priority)

    def solve_strat_zielonka(self):

        z = self.solve_value_zielonka()

        ret = np.full(len(self.owner), -1, dtype=int)
        edges = np.array(self.edges)

        for i,v in enumerate(edges):
            w = np.where(v)[0]
            while True:
                cl = ceil(len(w)/2)
                one,two = w[:cl],w[cl:]
                e = edges.copy()
                e[i]=False
                e[i,one]=True
                x = ParityGame(self.owner, e, self.priority).solve_value_zielonka()
                if np.all(x==z):
                    if len(one)==1:
                        ret[i]=one[0]
                        break
                    else:
                        w = one
                else:
                    w = two
            edges[i]=np.zeros(len(self.owner), dtype=bool)
            edges[i,ret[i]]=True
        return ret

    def solve_value(self, strat=None):

        if type(strat) != type(None):

            edges = (strat==-1).reshape(-1,1)*self.edges

            for i in np.where(strat!=-1)[0]:
                edges[i,strat[i]]=True

            return ParityGame(self.owner, edges, self.priority).solve_value_zielonka()

        else:

            return self.solve_value_zielonka()

    def solve_strat(self):

        return self.solve_strat_zielonka()

    def to_mpg(self):

        edges_exist = self.edges
        edges_value = np.fromfunction(lambda x,y: (-len(self.owner))**self.priority[x], shape=(len(self.owner),len(self.owner)), dtype=int)

        mini = np.iinfo(edges_value.dtype).min
        edges = np.where(edges_exist, edges_value, mini)

        return MeanPayoffGame(self.owner, edges)

    def visualise(self, target_path=None, strat=None, values=None):

        if type(strat) == type(None):
            strat = np.full(len(self.owner),-1)

        if target_path == None:
            target_path = os.path.join(gettempdir(), f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")

        view = Digraph(format="png")
        view.attr(bgcolor="#f0f0f0", forcelabels="True")
        for i,(owner, priority) in enumerate(zip(self.owner, self.priority)):
            if type(values) != type(None):
                label = f"<v(v<sub>{i}</sub>)={e_o[values[i]]}>"
            else:
                label = f"<v<sub>{i}</sub>>"
            view.node(f"{i}", label=label, xlabel=f"{priority}", shape=shape[owner], fontcolor=colour[owner][strat[i]!=-1], color=colour[owner][strat[i]!=-1])
        idx = np.where(self.edges==True)
        for s,t in zip(idx[0],idx[1]):
            view.edge(str(s),str(t), fontcolor=colour[self.owner[s]][strat[s]==t], color=colour[self.owner[s]][strat[s]==t])

        save_loc = view.render(filename=target_path, view=False, cleanup=True)

        return save_loc

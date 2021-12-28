from game import *
import numpy as np
from mpg import MeanPayoffGame
from tempfile import gettempdir
from graphviz import Digraph
from math import ceil
from itertools import count
from time import sleep

e_o = {0:"Even",1:"Odd"}

def find_attractor(player, owner, edges, Nh):

    if Nh.size==0:
        return []

    us = np.where(owner==player)[0]
    them = np.where(owner!=player)[0]
    
    atr = np.array(Nh)
    hashh = hash(atr.tobytes())

    while True:
        old = hashh
        us_add = np.any(edges[np.ix_(us,atr)], axis=1)
        them_add = np.logical_not(np.any(edges[np.ix_(them,np.setdiff1d(np.arange(owner.shape[0]),atr))], axis=1))
        atr = np.union1d(atr,us[us_add])
        atr = np.union1d(atr,them[them_add])
        hashh = hash(atr.tobytes())
        if hashh==old:
            break

    return atr

def zielonka(owner, edges, priority):

    if owner.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    else:
        m = np.max(priority)
        player = m%2
        U = np.where(priority==m)[0]
        A = find_attractor(player, owner, edges, U)
        A_inv = np.setdiff1d(np.arange(owner.shape[0]), A)
        even, odd = zielonka(owner[A_inv], edges[np.ix_(A_inv,A_inv)], priority[A_inv])
        if (player and even.size == 0) or (not player and odd.size == 0):
            return (np.array([], dtype=np.int64),np.union1d(A_inv[odd],A)) if player else (np.union1d(A_inv[even],A),np.array([], dtype=np.int64))
        else:
            B = find_attractor(1-player, owner, edges, A_inv[even] if player else A_inv[odd])
            B_inv = np.setdiff1d(np.arange(owner.shape[0]), B)
            even2, odd2 = zielonka(owner[B_inv], edges[np.ix_(B_inv,B_inv)], priority[B_inv])
            return (np.union1d(B_inv[even2],B),B_inv[odd2]) if player else (B_inv[even2],np.union1d(B_inv[odd2],B))

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

        z = zielonka(self.owner, self.edges, self.priority)

        ret = np.zeros(len(self.owner), dtype=bool)
        ret[z[1]] = True

        return ret

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
            label = f"<v(v<sub>{i}</sub>)={e_o[values[i]]}>"
            view.node(f"{i}", label=label, xlabel=f"{priority}", shape=shape[owner], fontcolor=colour[owner][strat[i]!=-1], color=colour[owner][strat[i]!=-1])
        idx = np.where(self.edges==True)
        for s,t in zip(idx[0],idx[1]):
            view.edge(str(s),str(t), fontcolor=colour[self.owner[s]][strat[s]==t], color=colour[self.owner[s]][strat[s]==t])

        save_loc = view.render(filename=target_path, view=False, cleanup=True)

        return save_loc

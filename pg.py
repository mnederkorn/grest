from game import *
import numpy as np
from mpg import MeanPayoffGame

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
        return np.array([], dtype=np.int64),np.array([], dtype=np.int64)
    else:
        m = np.max(priority)
        player = m%2
        U = np.where(priority==m)[0]
        A = find_attractor(player, owner, edges, U)
        A_inv = np.setdiff1d(np.arange(owner.shape[0]), A)
        # print(m,player,U,A,A_inv)
        even, odd = zielonka(owner[A_inv], edges[np.ix_(A_inv,A_inv)], priority[A_inv])
        # print(even, odd)
        if (player and even.size == 0) or (not player and odd.size == 0):
            return (np.array([], dtype=np.int64),np.union1d(A_inv[odd],A)) if player else (np.union1d(A_inv[even],A),np.array([], dtype=np.int64))
        else:
            B = find_attractor(1-player, owner, edges, A_inv[even] if player else A_inv[odd])
            B_inv = np.setdiff1d(np.arange(owner.shape[0]), B)
            # print("x",m,player,B,B_inv)
            even2, odd2 = zielonka(owner[B_inv], edges[np.ix_(B_inv,B_inv)], priority[B_inv])
            # print("x",even2, odd2)
            return (np.union1d(B_inv[even2],B),B_inv[odd2]) if player else (B_inv[even2],np.union1d(B_inv[odd2],B))

class ParityGame(Game):

    def __init__(self, owner, edges, priority):

        super().__init__(owner, edges)
        self.priority = priority

    @classmethod
    def generate(cls, n, p, h):
        if not h<=n: raise Error("h has to be smaller than n")
        owner = np.random.choice([False, True], size=(n))
        edges = np.empty((n,n), dtype=bool)
        for e in edges:
            rng = np.random.randint(n)
            e[rng] = True
            e[:rng] = np.random.choice([False, True], size=rng, p=[1-p, p])
            e[rng+1:] = np.random.choice([False, True], size=n-(rng+1), p=[1-p, p])
        priority = np.random.randint(0, h, size=(n))

        return cls(owner, edges, priority)

    def solve_zielonka(self):

        return zielonka(self.owner, self.edges, self.priority)

    def to_mpg(self):

        edges_exist = self.edges
        edges_value = np.fromfunction(lambda x,y: (-len(self.owner))**self.priority[x], shape=(len(self.owner),len(self.owner)), dtype=int)

        mini = np.iinfo(edges_value.dtype).min
        edges = np.where(edges_exist, edges_value, mini)

        return MeanPayoffGame(self.owner, edges, np.array((0,)))

    def visualise(self, target_path=None):

        if target_path == None:
            target_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images", f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}")

        view = Digraph(format="png")
        for i,(owner, priority) in enumerate(zip(self.owner, self.priority)):
            if not owner:
                view.node(str(i), str(priority)+"_"+str(i), shape="diamond", fontcolor="#000000")
            else:
                view.node(str(i), str(priority)+"_"+str(i), shape="square", fontcolor="#000000")
        idx = np.where(self.edges==True)
        for s,t in zip(idx[0],idx[1]):
            view.edge(str(s),str(t))
        view.render(filename=target_path, view=False, cleanup=True)

        return target_path+r".png"
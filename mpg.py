from game import *
import numpy as np
from math import floor,ceil
from fractions import Fraction
from dpg import DiscountedPayoffGame
from eg import EnergyGame
from graphviz import Digraph

class MeanPayoffGame(Game):

    def __init__(self, owner, edges, threshold):

        super().__init__(owner, edges)
        self.threshold = threshold

    @classmethod
    def generate(cls, n, p, w):
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
        threshold = np.random.randint(-w*n,(w*n)+1, size=((1,)))

        return cls(owner, edges, threshold)

    def solve_zwick_paterson(self):

        mini = np.iinfo(self.edges.dtype).min

        empty_as_zero = np.where(self.edges != mini, self.edges, 0)

        W = max(abs(np.amin(empty_as_zero)),abs(np.amax(empty_as_zero)))

        a = np.zeros(len(self.owner)).reshape((-1,1))

        k = (4*(len(self.owner)**3)*W)

        for x in range(k):

            edges_weight = np.where(self.edges != mini, self.edges+a, np.nan)

            a = np.where(self.owner, np.nanmin(edges_weight, 1), np.nanmax(edges_weight, 1))

        lower = (a/k)-(1/(2*len(self.owner)*(len(self.owner)-1)))
        upper = (a/k)+(1/(2*len(self.owner)*(len(self.owner)-1)))

        v = np.zeros((len(self.owner)),dtype=object)

        for v_n in range(len(self.owner)):
            br=False
            for denominator in range(1,len(self.owner)+1):
                if br: break
                f=floor(lower[v_n]*denominator)
                c=ceil(upper[v_n]*denominator)
                for numerator in range(f,c+1):
                    r=Fraction(numerator,denominator)
                    if ((lower[v_n]<=r)&(r<=upper[v_n])):
                        v[v_n]=r
                        br = True
                        break

        return v

    def to_dpg(self):

        mini = np.iinfo(self.edges.dtype).min

        empty_as_zero = np.where(self.edges != mini, self.edges, 0)

        W = max(abs(np.amin(empty_as_zero)),abs(np.amax(empty_as_zero)))

        discount = 1-(1/(4*(len(self.owner)**3)*W))

        return DiscountedPayoffGame(self.owner, self.edges, self.threshold, np.array((discount,)))

    def to_eg(self):

        return EnergyGame(self.owner, self.edges, np.array((0,)))

    def visualise(self, target_path=None, strat=None):

        if target_path == None:
            target_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images", f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}")

        view = Digraph(format="png")
        view.attr(bgcolor="#f0f0f0")
        for i,owner in enumerate(self.owner):
            if not owner:
                if strat[i]!=-1:
                    view.node(str(i), shape="square", fontcolor="#00df00", color="#00df00")
                else:
                    view.node(str(i), shape="square", fontcolor="#003f00", color="#003f00")
            else:
                if strat[i]!=-1:
                    view.node(str(i), shape="circle", fontcolor="#df0000", color="#df0000")
                else:
                    view.node(str(i), shape="circle", fontcolor="#3f0000", color="#3f0000")
        mini = np.iinfo(self.edges.dtype).min
        idx = np.where(self.edges!=mini)
        for s,t in zip(idx[0],idx[1]):
            if self.owner[s]:
                if strat[s]==t:
                    view.edge(str(s),str(t),str(self.edges[s,t]), fontcolor="#df0000", color="#df0000")
                else:
                    view.edge(str(s),str(t),str(self.edges[s,t]), fontcolor="#3f0000", color="#3f0000")
            else:
                if strat[s]==t:
                    view.edge(str(s),str(t),str(self.edges[s,t]), fontcolor="#00df00", color="#00df00")
                else:
                    view.edge(str(s),str(t),str(self.edges[s,t]), fontcolor="#003f00", color="#003f00")
        view.render(filename=target_path, view=False, cleanup=True)

        return target_path+r".png"

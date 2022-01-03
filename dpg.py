from game import *
import numpy as np
from ortools.linear_solver import pywraplp
from ssg import SimpleStochasticGame
from graphviz import Digraph
from tempfile import gettempdir
import copy
from numba import jit

@jit(nopython=True, cache=True)
def numba_nanmin_axis1(x):
    out = np.empty(x.shape[0], dtype=x.dtype)
    for i in range(x.shape[0]):
        out[i] = np.nanmin(x[i])
    return out

@jit(nopython=True, cache=True)
def numba_nanmax_axis1(x):
    out = np.empty(x.shape[0], dtype=x.dtype)
    for i in range(x.shape[0]):
        out[i] = np.nanmax(x[i])
    return out

@jit(nopython=True, cache=True)
def numba_argmax_axis1(x):
    out = np.empty(x.shape[0], dtype=np.int32)
    for i in range(x.shape[1]):
        out[i] = np.argmax(x[i])
    return out

@jit(nopython=True, cache=True)
def numba_argmin_axis1(x):
    out = np.empty(x.shape[0], dtype=np.int32)
    for i in range(x.shape[1]):
        out[i] = np.argmin(x[i])
    return out

class DiscountedPayoffGame(Game):

    def __init__(self, owner, edges, discount):

        super().__init__(owner, edges)
        self.discount = discount

    @classmethod
    def generate(cls, n, p, w):

        return cls(*super().generate(n, p, w), np.random.rand(1))

    def to_ssg(self):

        mini = np.iinfo(self.edges.dtype).min

        W = np.max(np.abs(np.where(self.edges!=mini, self.edges, 0)))

        edges = np.where(self.edges != mini, self.edges+W, self.edges)

        f3 = np.count_nonzero(self.edges != mini)

        vertices = len(self.owner) + f3

        f3pos = np.where(self.edges != mini)

        ssg_edges = np.full((vertices,vertices+2), False)

        owner = np.hstack((self.owner, np.full(f3, 2, dtype=np.uint8)))

        avg_chance = np.zeros((f3, vertices+2))

        strat_map = np.zeros(f3, dtype=int)

        for i,edge in enumerate(zip(f3pos[0],f3pos[1])):
            ssg_edges[edge[0], i+len(self.owner)] = True
            ssg_edges[i+len(self.owner), edge[1]] = True
            strat_map[i]=edge[1]
            ssg_edges[i+len(self.owner), -1] = True
            ssg_edges[i+len(self.owner), -2] = True
            avg_chance[i, edge[1]] = self.discount
            avg_chance[i, -2] = (1-self.discount)*(1-(edges[edge]/(2*W)))
            avg_chance[i, -1] = (1-self.discount)*(edges[edge]/(2*W))
            
        return SimpleStochasticGame(owner, ssg_edges, avg_chance, True), strat_map, W

    def solve_both_ssg(self):

        ssg, smap, W = self.to_ssg()

        v, s = ssg.solve_both_kleene()

        v = (v[:len(self.owner)]*2*W)-W

        s = smap[s-len(self.owner)][:len(self.owner)]

        return v, s

    def solve_both_kleene_wrap(self):

        return self.solve_both_kleene(self.owner, self.edges, self.discount)

    @staticmethod
    @jit(nopython=True, cache=True)
    def solve_both_kleene(owner, edges, discount):

        mini = np.iinfo(edges.dtype).min
        maxi = np.iinfo(edges.dtype).max

        cur = np.zeros(len(owner))

        while True:

            old = cur.copy()

            edges_weight = np.where(edges != mini, ((1-discount)*edges)+discount*cur, np.nan)

            cur = np.where(owner, numba_nanmin_axis1(edges_weight), numba_nanmax_axis1(edges_weight))

            max_err = np.amax(np.abs(cur-old))

            # iterate until max float precision is hit
            if max_err <1e-14:
                break

        strat = np.where(owner, numba_argmin_axis1(np.where(edges!=mini, (1-discount)*edges+discount*cur, maxi)), numba_argmax_axis1(np.where(edges!=mini, (1-discount)*edges+discount*cur, mini)))

        print(strat.dtype)

        return cur, strat

    def solve_both_strat_iter(self, player):

        p0 = np.where(self.owner==False)[0]
        p1 = np.where(self.owner==True)[0]

        mini = np.iinfo(self.edges.dtype).min
        maxi = np.iinfo(self.edges.dtype).max

        if not player:
            strat = np.where(self.owner, -1, np.apply_along_axis(lambda x: np.random.choice(np.where(x!=mini)[0]), 1, self.edges))
        else:
            strat = np.where(self.owner, np.apply_along_axis(lambda x: np.random.choice(np.where(x!=mini)[0]), 1, self.edges), -1)

        strat_hist = []

        while not hash(strat.tobytes()) in strat_hist:

            strat_hist.append(hash(strat.tobytes()))

            weights = self.edges[np.where(self.edges!=mini)]
            W = max(abs(np.amin(weights)),abs(np.amax(weights)))

            solver = pywraplp.Solver.CreateSolver('GLOP')

            v = [solver.NumVar(float(-W), float(W), str(x)) for x in range(len(self.owner))]

            if not player:
                for s,p in enumerate(self.owner):
                    if not p:
                        solver.Add(v[s] == (1-float(self.discount))*float(self.edges[s,strat[s]])+float(self.discount)*v[strat[s]])
                    else:
                        for t in np.where(self.edges[s]!=mini)[0]:
                            solver.Add(v[s] <= (1-float(self.discount))*float(self.edges[s,t])+float(self.discount)*v[t])
            else:
                for s,p in enumerate(self.owner):
                    if p:
                        solver.Add(v[s] == (1-float(self.discount))*float(self.edges[s,strat[s]])+float(self.discount)*v[strat[s]])
                    else:
                        for t in np.where(self.edges[s]!=mini)[0]:
                            solver.Add(v[s] >= (1-float(self.discount))*float(self.edges[s,t])+float(self.discount)*v[t])

            obj_func = v[0]

            for v_n in v[1:]:
                obj_func+=v_n

            if not player:
                solver.Maximize(obj_func)
            else:
                solver.Minimize(obj_func)

            status = solver.Solve()

            if not player:
                strat = np.where(self.owner, strat, np.argmax(((1-self.discount)*self.edges)+(self.discount*(np.array([v_n.solution_value() for v_n in v]))), 1))
            else:
                strat = np.where(self.owner, np.argmin(((1-self.discount)*np.where(self.edges==mini,maxi,self.edges))+(self.discount*(np.array([v_n.solution_value() for v_n in v]))), 1), strat)

        return np.array([v_n.solution_value() for v_n in v]),strat

    def solve_value(self, strat=None):

        if type(strat) != type(None):

            mini = np.iinfo(self.edges.dtype).min

            edges = np.where((strat==-1).reshape(-1,1), self.edges, mini)

            for i in np.where(strat!=-1)[0]:
                edges[i,strat[i]]=self.edges[i,strat[i]]

            return DiscountedPayoffGame(self.owner, edges, self.discount).solve_both_kleene_wrap()[0]

        else:

            return self.solve_both_kleene_wrap()[0]

    def solve_strat(self):

            return self.solve_both_kleene_wrap()[1]

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
        idx = np.where(self.edges!=mini)
        for s,t in zip(idx[0],idx[1]):
            view.edge(str(s),str(t),str(self.edges[s,t]), fontcolor=colour[self.owner[s]][strat[s]==t], color=colour[self.owner[s]][strat[s]==t])
                    
        save_loc = view.render(filename=target_path, view=False, cleanup=True)

        return save_loc

from game import *
import numpy as np
# from ortools.linear_solver import pywraplp
from tempfile import gettempdir
from graphviz import Digraph
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

class SimpleStochasticGame_n(Game):

    def __init__(self, owner, edges, avg_chance, stopping):

        super().__init__(owner, edges)
        self.avg_chance = avg_chance
        self.stopping = stopping

    @classmethod
    def generate(cls, n, p):
        assert p>=1/n, "Since |post(v)| needs to be >=1 for every v, p needs to be at least p>=1/n"
        p=((p*n)-1)/(n-1)
        owner = np.random.randint(0, 3, size=(n), dtype=np.uint8)
        edges = np.empty((n,n+2), dtype=bool)
        for e in edges:
            rng = np.random.randint(n+2)
            e[rng] = True
            e[:rng] = np.random.choice([False, True], size=rng, p=[1-p, p])
            e[rng+1:] = np.random.choice([False, True], size=(n+2)-(rng+1), p=[1-p, p])

        avg_n = np.count_nonzero(owner==2)
        avg_chance = np.random.random(size=(avg_n,n+2))
        avg_chance = np.where(edges[np.where(owner==2)], avg_chance, 0)

        avg_chance = avg_chance/np.sum(avg_chance,1).reshape(-1,1)

        return cls(owner, edges, avg_chance, False)

    def solve_value_kleene_wrap(self):

        return self.solve_value_kleene(self.owner, self.edges, self.avg_chance, self.stopping)

    @staticmethod
    @jit(nopython=True, cache=True)
    def solve_value_kleene(owner, edges, avg_chance, stopping):

        cur = np.hstack((np.zeros(len(owner)), np.array([0]), np.array([1])))

        while True:

            old = cur.copy()

            # edges_weight = np.tile(cur, (len(owner),1))
            edges_weight = cur.repeat(len(owner)).reshape(-1,len(owner)).transpose()
            edges_weight = np.where(edges, edges_weight, np.nan)

            cur[:-2] = np.where(owner == 0, numba_nanmax_axis1(edges_weight), cur[:-2])
            cur[:-2] = np.where(owner == 1, numba_nanmin_axis1(edges_weight), cur[:-2])

            idx = np.where(owner == 2)
            cur[idx]=np.sum(avg_chance*old, 1)

            max_err = np.amax(np.abs(cur-old))

            # iterate until max float precision is hit
            if max_err == 0:
                break

        return cur

    def solve_both_strat_iter(self, player):

        assert self.stopping, "SSG needs to be stopping to be solved with strategy iteration. To ensure SSG is stopping, generate DPG and convert to SSG via DiscountedPayoffGame.to_ssg."

        p0 = np.where(self.owner==0)[0]
        p1 = np.where(self.owner==1)[0]
        p2 = np.where(self.owner==2)[0]

        if not player:
            strat = np.where(self.owner==0, np.apply_along_axis(lambda x: np.random.choice(np.where(x)[0]), 1, self.edges), -1)
        else:
            strat = np.where(self.owner==1, np.apply_along_axis(lambda x: np.random.choice(np.where(x)[0]), 1, self.edges), -1)

        strat_hist = []

        while not hash(strat.tobytes()) in strat_hist:

            strat_hist.append(hash(strat.tobytes()))

            solver = pywraplp.Solver.CreateSolver('GLOP')

            v = [solver.NumVar(float(0), float(1), str(x)) for x in range(len(self.owner))]+[solver.NumVar(float(0), float(0), str(len(self.owner)+1))]+[solver.NumVar(float(1), float(1), str(len(self.owner)+2))]

            for s,p in enumerate(self.owner):
                if p==player:
                    solver.Add(v[s] == v[strat[s]])
                else:
                    if p==0:
                        for t in np.where(self.edges[s])[0]:
                            solver.Add(v[s] >= v[t])
                    elif p==1:
                        for t in np.where(self.edges[s])[0]:
                            solver.Add(v[s] <= v[t])
                    else:
                        val = 0
                        for t in np.where(self.edges[s])[0]:
                            val += (v[t]*self.avg_chance[np.nonzero(s==p2)[0][0],t])
                        solver.Add(v[s] == val)

            obj_func = v[0]

            for v_n in v[1:]:
                obj_func+=v_n

            if not player:
                solver.Maximize(obj_func)
            else:
                solver.Minimize(obj_func)

            status = solver.Solve()

            if not player:
                strat = np.where(self.owner==0, np.argmax(np.where(self.edges, self.edges*np.array([v_n.solution_value() for v_n in v]), -1), 1), strat)
            else:
                strat = np.where(self.owner==1, np.argmin(np.where(self.edges, self.edges*np.array([v_n.solution_value() for v_n in v]), 2), 1), strat)

        return np.array([v_n.solution_value() for v_n in v]), strat

    # strats for avg/rng vertices as -1
    def solve_strat_kleene(self):

        z = self.solve_value_kleene()

        strats = np.where(self.owner==2, -1, np.where(self.owner==0, np.nanargmax(np.where(self.edges, z, np.nan), 1), np.nanargmin(np.where(self.edges, z, np.nan), 1)))

        return strats

    def solve_value(self, strat=None):

        print(self)

        if type(strat) != type(None):

            edges = (strat==-1).reshape(-1,1)*self.edges

            for i in np.where(strat!=-1)[0]:
                edges[i,strat[i]]=True

            return SimpleStochasticGame(self.owner, edges, self.avg_chance, self.stopping).solve_value_kleene()

        else:

            return self.solve_value_kleene()

    def solve_strat(self):

        return self.solve_strat_kleene()

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

        view.node(str(len(self.owner)), label="0", shape=shape[2], fontcolor=colour[2][False], color=colour[2][False], peripheries="2")
        view.node(str(len(self.owner)+1), label="1", shape=shape[2], fontcolor=colour[2][False], color=colour[2][False], peripheries="2")

        idx = np.where(self.edges==True)
        for s,t in zip(idx[0],idx[1]):
            if self.owner[s]!=2:
                view.edge(str(s),str(t), fontcolor=colour[self.owner[s]][strat[s]==t], color=colour[self.owner[s]][strat[s]==t])

        for i,s in enumerate(np.where(self.owner==2)[0]):
            for t in np.where(self.edges[s]==True)[0]:
                view.edge(str(s),str(t),f"{self.avg_chance[i,t]:.2f}",fontcolor=colour[2][False], color=colour[2][False])

        save_loc = view.render(filename=target_path, view=False, cleanup=True)

        return save_loc


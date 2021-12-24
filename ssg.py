from game import *
import numpy as np
from ortools.linear_solver import pywraplp
from tempfile import gettempdir
from graphviz import Digraph

class SimpleStochasticGame(Game):

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

        avg_chance = (avg_chance.transpose()/np.sum(avg_chance,1)).transpose()

        return cls(owner, edges, avg_chance, False)

    def solve_value_iter(self):

        cur = np.hstack((np.zeros(len(self.owner)), [0], [1]))

        while True:

            old = np.array(cur)

            edges_weight = np.tile(cur, (len(self.owner),1))
            edges_weight = np.where(self.edges, edges_weight, np.nan)

            cur[:-2] = np.where(self.owner == 0, np.nanmax(edges_weight, 1), cur[:-2])
            cur[:-2] = np.where(self.owner == 1, np.nanmin(edges_weight, 1), cur[:-2])

            idx = np.where(self.owner == 2)
            cur[idx]=np.sum(self.avg_chance*old, 1)

            max_err = np.amax(np.abs(cur-old))

            # iterate until max float precision is hit
            if max_err == 0:
                break

        return cur

    def solve_strat_iter(self):

        assert self.stopping, "SSG needs to be stopping to be solved with strategy iteration. To ensure SSG is stopping, generate DPG and convert to SSG via DiscountedPayoffGame.to_ssg."

        p0 = np.where(self.owner==0)[0]
        p1 = np.where(self.owner==1)[0]
        p2 = np.where(self.owner==2)[0]

        if len(p0) != 0:
            rnd_strat = np.apply_along_axis(lambda x: np.random.choice(np.where(x)[0]), 1, self.edges[p0])
        else:
            rnd_strat = np.array([])

        rnd_strat = np.vstack((p0,rnd_strat))

        rnd_strat = np.transpose(rnd_strat)

        strat = rnd_strat

        strat_hist = []

        while not hash(strat.tobytes()) in strat_hist:

            strat_hist.append(hash(strat.tobytes()))

            solver = pywraplp.Solver.CreateSolver('GLOP')

            v = [solver.NumVar(float(0), float(1), str(x)) for x in range(len(self.owner))]+[solver.NumVar(float(0), float(0), str(len(self.owner)+1))]+[solver.NumVar(float(1), float(1), str(len(self.owner)+2))]

            for s,t in strat:
                solver.Add(v[s] == v[t])

            for s in p1:
                for t in np.where(self.edges[s])[0]:
                    solver.Add(v[s] <= v[t])

            for i,s in enumerate(p2):
                val = sum([v[val_n]*self.avg_chance[i,val_n] for val_n in np.where(self.edges[s])[0]])
                solver.Add(v[s] == val)

            obj_func = v[0]

            for v_n in v[1:]:
                obj_func+=v_n

            solver.Maximize(obj_func)

            status = solver.Solve()

            strat = np.empty(0, dtype=np.int64)

            for s in p0:
                idx = np.where(self.edges[s])[0]
                new = np.argmax([v[t].solution_value() for t in idx])
                strat = np.hstack((strat, idx[new]))

            strat = np.transpose(np.vstack((p0,strat)))

        return np.array([v_n.solution_value() for v_n in v])

    def solve(self, strat=None):

        if type(strat) != type(None):

            old = np.array(self.edges)

            self.edges = np.where(strat!=-1, 0, self.edges.transpose()).transpose()

            for i in np.where(strat!=-1)[0]:
                self.edges[i,strat[i]]=old[i,strat[i]]

            ret = self.solve_value_iter()

            self.edges = old

        else:

            ret = self.solve_value_iter()

        return ret

    # strats for avg/rng vertices as -1
    def solve_strat_value_iter(self):

        z = self.solve_value_iter()

        strats = np.where(self.owner==2, -1, np.where(self.owner==0, np.nanargmax(np.where(self.edges, z, np.nan), 1), np.nanargmin(np.where(self.edges, z, np.nan), 1)))

        return strats

    def visualise(self, target_path=None, strat=None, values=None, restr_values=None):

        if type(strat) == type(None):
            strat = np.full(self.owner.shape[0],-1)

        if target_path == None:
            target_path = os.path.join(gettempdir(), f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")

        view = Digraph(format="png")
        view.attr(bgcolor="#f0f0f0")
        for i,owner in enumerate(self.owner):
            if (type(values) == type(None)) and (type(restr_values) == type(None)):
                label = f"<v<sub>{i}</sub>>"
            elif (type(values) != type(None)) and (type(restr_values) == type(None)):
                label = f"<v(v<sub>{i}</sub>)={float(values[i]):.2f}>"
            elif (type(values) == type(None)) and (type(restr_values) != type(None)):
                label = f"<v<sub>|</sub>(v<sub>{i}</sub>)={float(restr_values[i]):.2f}>"
            else:
                label = f"<v(v<sub>{i}</sub>)={float(values[i]):.2f}<br/>v<sub>|</sub>(v<sub>{i}</sub>)={float(restr_values[i]):.2f}>"
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


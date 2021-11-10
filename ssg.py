from game import *
import numpy as np
from ortools.linear_solver import pywraplp

class SimpleStochasticGame(Game):

    def __init__(self, owner, edges, avg_chance):

        super().__init__(owner, edges)
        self.avg_chance = avg_chance

    @classmethod
    def generate(cls, n, p):
        owner = np.random.randint(0, 3, size=(n), dtype=np.uint8)
        edges = np.empty((n,n+2), dtype=bool)
        for e in edges:
            rng = np.random.randint(n+2)
            e[rng] = True
            e[:rng] = np.random.choice([False, True], size=rng, p=[1-p, p])
            e[rng+1:] = np.random.choice([False, True], size=(n+2)-(rng+1), p=[1-p, p])

        avg_n = np.count_nonzero(owner==2)
        avg_chance = np.random.randint(0, 256, size=(avg_n,n+2), dtype=np.uint32)
        avg_chance = np.where(edges[np.where(owner==2)], avg_chance, 0)

        return cls(owner, edges, avg_chance)

    def solve_value_iter(self):

        cur = np.hstack((np.zeros(len(self.owner)), [0], [1]))

        while True:

            old = np.array(cur)

            edges_weight = np.tile(cur, (len(self.owner),1))
            edges_weight = np.where(self.edges, edges_weight, np.nan)

            cur[:-2] = np.where(self.owner == 0, np.nanmax(edges_weight, 1), cur[:-2])
            cur[:-2] = np.where(self.owner == 1, np.nanmin(edges_weight, 1), cur[:-2])

            idx = np.where(self.owner == 2)
            cur[idx]=np.sum(self.avg_chance*old, 1)/np.sum(self.avg_chance, 1)

            max_err = np.amax(cur-old)

            # iterate until max float precision is hit
            if max_err == 0:
                break

        return cur

    def solve_strat_iter(self): 

        p0 = np.where(self.owner==0)[0]
        p1 = np.where(self.owner==1)[0]
        p2 = np.where(self.owner==2)[0]

        rnd_strat = np.apply_along_axis(lambda x: np.random.choice(np.where(x)[0]), 1, self.edges[p0])

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
                where = np.where(self.edges[s])[0]
                su = np.sum(self.avg_chance[i,where])
                val = sum([v[val_n]*(self.avg_chance[i,val_n]/su) for val_n in where])
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

    def visualise(self, target_path=None):

        # colour="#dfdfdf"
        colour="#000000"

        if target_path == None:
            target_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images", f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}")

        view = Digraph(format="png")
        for i,owner in enumerate(self.owner):
            if owner==0:
                view.node(str(i), shape="square", fontcolor=colour)
            elif owner==1:
                view.node(str(i), shape="circle", fontcolor=colour)
            elif owner==2:                
                view.node(str(i), shape="diamond", fontcolor=colour)
        view.node(str(len(self.owner)), label="MIN", shape="diamond", peripheries="2")
        view.node(str(len(self.owner)+1), label="MAX", shape="diamond", peripheries="2")

        for s in np.where(self.owner!=2)[0]:
            for t in np.where(self.edges[s]==True)[0]:
                view.edge(str(s),str(t))

        for i,s in enumerate(np.where(self.owner==2)[0]):
            total = np.sum(self.avg_chance[i])
            for t in np.where(self.edges[s]==True)[0]:
                view.edge(str(s),str(t),f"{self.avg_chance[i,t]}/{total}")

        view.render(filename=target_path, view=False, cleanup=True)

        return target_path+r".png"

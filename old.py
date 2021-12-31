from game import *
import numpy as np
from ortools.linear_solver import pywraplp
from ssg import SimpleStochasticGame
from graphviz import Digraph
from tempfile import gettempdir
import copy
from numba import jit

class DiscountedPayoffGame_n(Game):

    def __init__(self, owner, edges, discount):

        super().__init__(owner, edges)
        self.discount = discount

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
        discount = np.random.rand(1)

        return cls(owner, edges, discount)

    def to_ssg(self):

        mini = np.iinfo(self.edges.dtype).min

        empty_as_zero = np.where(self.edges != mini, self.edges, 0)

        W = max(abs(np.amin(empty_as_zero)),abs(np.amax(empty_as_zero)))

        edges = np.where(self.edges != mini, self.edges+W, self.edges)

        f3 = np.count_nonzero(self.edges != mini)

        vertices = len(self.owner) + f3

        f3pos = np.where(self.edges != mini)

        ssg_edges = np.full((vertices,vertices+2), False)

        owner = np.hstack((self.owner, np.full(f3, 2, dtype=np.uint8)))

        avg_chance = np.zeros((f3, vertices+2))

        for i,edge in enumerate(zip(f3pos[0],f3pos[1])):
            ssg_edges[edge[0], i+len(self.owner)] = True
            ssg_edges[i+len(self.owner), edge[1]] = True
            ssg_edges[i+len(self.owner), -1] = True
            ssg_edges[i+len(self.owner), -2] = True
            avg_chance[i, edge[1]] = self.discount
            avg_chance[i, -2] = (1-self.discount)*(1-(edges[edge]/(2*W)))
            avg_chance[i, -1] = (1-self.discount)*(edges[edge]/(2*W))
            
        return SimpleStochasticGame(owner, ssg_edges, avg_chance)

    def solve_value_kleene_wrapper(self):

        return self.solve_value_kleene(self.owner,self.edges,self.discount)

    @staticmethod
    @jit(nopython=True)
    def solve_value_kleene(owner,edges,discount):

        mini = np.iinfo(edges.dtype).min

        cur = np.zeros(len(owner))

        while True:

            old = cur.copy()

            edges_weight = np.where(edges != mini, ((1-discount)*edges)+discount*cur, np.nan)

            # cur = np.where(owner, np.nanmin(edges_weight, 1), np.nanmax(edges_weight, 1))

            cur = np.where(owner, np.array([np.nanmin(i) for i in edges_weight]), np.array([np.nanmax(i) for i in edges_weight]))

            # print(edges_weight)

            max_err = np.amax(np.abs(cur-old))

            # iterate until max float precision is hit
            if max_err <1e-14:
                break

        return cur

    def solve_strat_iter(self):        

        p0 = np.where(self.owner==False)[0]
        p1 = np.where(self.owner==True)[0]

        mini = np.iinfo(self.edges.dtype).min

        rnd_strat = np.apply_along_axis(lambda x: np.random.choice(np.where(x!=mini)[0]), 1, self.edges[p0])

        rnd_strat = np.vstack((p0,rnd_strat))

        rnd_strat = np.transpose(rnd_strat)

        strat = rnd_strat

        strat_hist = []

        while not hash(strat.tobytes()) in strat_hist:

            strat_hist.append(hash(strat.tobytes()))

            weights = self.edges[np.where(self.edges!=mini)]
            W = max(abs(np.amin(weights)),abs(np.amax(weights)))

            solver = pywraplp.Solver.CreateSolver('GLOP')

            v = [solver.NumVar(float(-W), float(W), str(x)) for x in range(len(self.owner))]

            for s,t in strat:
                solver.Add(v[s] == (1-float(self.discount))*float(self.edges[s,t])+float(self.discount)*v[t])

            for s in p1:
                for t in np.where(self.edges[s]!=mini)[0]:
                    solver.Add(v[s] <= (1-float(self.discount))*float(self.edges[s,t])+float(self.discount)*v[t])

            obj_func = v[0]

            for v_n in v[1:]:
                obj_func+=v_n

            solver.Maximize(obj_func)

            status = solver.Solve()

            strat = np.empty(0, dtype=np.int64)

            for s in p0:
                idx = np.where(self.edges[s]!=mini)[0]
                new = np.argmax([(1-self.discount)*self.edges[s,t]+self.discount*v[t].solution_value() for t in idx])
                strat = np.hstack((strat, idx[new]))

            strat = np.transpose(np.vstack((p0,strat)))

        return np.array([v_n.solution_value() for v_n in v])

    def solve(self, strat=None):

        if type(strat) != type(None):

            mini = np.iinfo(self.edges.dtype).min

            old = np.array(self.edges)

            self.edges = np.where(strat!=-1, mini, self.edges.transpose()).transpose()

            for i in np.where(strat!=-1)[0]:
                self.edges[i,strat[i]]=old[i,strat[i]]

            ret = self.solve_value_iter()

            self.edges = old

        else:

            ret = self.solve_value_iter()

        return ret

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
        mini = np.iinfo(self.edges.dtype).min
        idx = np.where(self.edges!=mini)
        for s,t in zip(idx[0],idx[1]):
            view.edge(str(s),str(t),str(self.edges[s,t]), fontcolor=colour[self.owner[s]][strat[s]==t], color=colour[self.owner[s]][strat[s]==t])
                    
        save_loc = view.render(filename=target_path, view=False, cleanup=True)

        return save_loc

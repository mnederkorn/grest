from game import *
import numpy as np
from ortools.linear_solver import pywraplp
from ssg import SimpleStochasticGame
from graphviz import Digraph
from tempfile import gettempdir

class DiscountedPayoffGame(Game):

    def __init__(self, owner, edges, threshold, discount):

        super().__init__(owner, edges)
        self.threshold = threshold
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
        threshold = np.random.randint(-w*n,(w*n)+1, size=((1,)))
        discount = np.random.rand(1)

        return cls(owner, edges, threshold, discount)

    def to_ssg(self):

        mini = np.iinfo(self.edges.dtype).min

        empty_as_zero = np.where(self.edges != mini, self.edges, 0)

        W = max(abs(np.amin(empty_as_zero)),abs(np.amax(empty_as_zero)))

        edges = np.where(self.edges != mini, self.edges+W, self.edges)

        f3 = np.count_nonzero(self.edges != mini)

        vertices = len(self.owner) + f3

        f3pos = np.where(self.edges != mini)

        denominator = np.iinfo(np.uint32).max

        ssg_edges = np.full((vertices,vertices+2), False)

        owner = np.hstack((self.owner, np.full(f3, 2, dtype=np.uint8)))

        avg_chance = np.zeros((f3, vertices+2), dtype=np.uint32)

        for i,edge in enumerate(zip(f3pos[0],f3pos[1])):
            ssg_edges[edge[0], i+len(self.owner)] = True
            ssg_edges[i+len(self.owner), edge[1]] = True
            ssg_edges[i+len(self.owner), -1] = True
            ssg_edges[i+len(self.owner), -2] = True
            avg_chance[i, edge[1]] = int(denominator*self.discount)
            avg_chance[i, -2] = int(denominator*(1-self.discount)*(1-(edges[edge]/(2*W))))
            avg_chance[i, -1] = int(denominator*(1-self.discount)*(edges[edge]/(2*W)))

        return SimpleStochasticGame(owner, ssg_edges, avg_chance), W

    def solve_value_iter(self):

        mini = np.iinfo(self.edges.dtype).min

        cur = np.zeros(len(self.owner)).reshape((-1,1))

        while True:

            old = np.array(cur)

            edges_weight = np.where(self.edges != mini, ((1-self.discount)*self.edges)+self.discount*cur, np.nan)

            cur = np.where(self.owner, np.nanmin(edges_weight, 1), np.nanmax(edges_weight, 1))

            max_err = np.amax(cur-old)

            # iterate until max float precision is hit
            if max_err == 0:
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

    def visualise(self, target_path=None, strat=None):

        if target_path == None:
            target_path = os.path.join(gettempdir(), f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}")

        view = Digraph(format="png")
        view.attr(bgcolor="#f0f0f0")
        for i,owner in enumerate(self.owner):
            if not owner:
                if strat[i]!=-1:
                    view.node(str(i), shape="square", fontcolor=colour["green"]["bright"], color=colour["green"]["bright"])
                else:
                    view.node(str(i), shape="square", fontcolor=colour["green"]["dark"], color=colour["green"]["dark"])
            else:
                if strat[i]!=-1:
                    view.node(str(i), shape="circle", fontcolor=colour["red"]["bright"], color=colour["red"]["bright"])
                else:
                    view.node(str(i), shape="circle", fontcolor=colour["red"]["dark"], color=colour["red"]["dark"])
        mini = np.iinfo(self.edges.dtype).min
        idx = np.where(self.edges!=mini)
        for s,t in zip(idx[0],idx[1]):
            if not self.owner[s]:
                if strat[s]==t:
                    view.edge(str(s),str(t),str(self.edges[s,t]), fontcolor=colour["green"]["bright"], color=colour["green"]["bright"])
                else:
                    view.edge(str(s),str(t),str(self.edges[s,t]), fontcolor=colour["green"]["dark"], color=colour["green"]["dark"])
            else:
                if strat[s]==t:
                    view.edge(str(s),str(t),str(self.edges[s,t]), fontcolor=colour["red"]["bright"], color=colour["red"]["bright"])
                else:
                    view.edge(str(s),str(t),str(self.edges[s,t]), fontcolor=colour["red"]["dark"], color=colour["red"]["dark"])
                    
        save_loc = view.render(filename=target_path, view=False, cleanup=True)

        return save_loc

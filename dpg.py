from game import *
import numpy as np
from ortools.linear_solver import pywraplp
from ssg import SimpleStochasticGame
from graphviz import Digraph
from tempfile import gettempdir
import copy

class DiscountedPayoffGame(Game):

    def __init__(self, owner, edges, discount):

        super().__init__(owner, edges)
        self.discount = discount

    @classmethod
    def generate(cls, n, p, w):
        assert p>=1/n, "Since |post(v)| needs to be >=1 for every v, p needs to be at least p>=1/n"
        p=((p*n)-1)/(n-1)
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

        # to_ssg_strat = np.argmax(ssg.edges[strat[np.where(strat!=-1)],:-2], 1)

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

        strat_map = np.zeros((len(self.owner),len(self.owner)), dtype=bool)

        for i,edge in enumerate(zip(f3pos[0],f3pos[1])):
            ssg_edges[edge[0], i+len(self.owner)] = True
            ssg_edges[i+len(self.owner), edge[1]] = True
            ssg_edges[i+len(self.owner), -1] = True
            ssg_edges[i+len(self.owner), -2] = True
            avg_chance[i, edge[1]] = self.discount
            avg_chance[i, -2] = (1-self.discount)*(1-(edges[edge]/(2*W)))
            avg_chance[i, -1] = (1-self.discount)*(edges[edge]/(2*W))
            
        return SimpleStochasticGame(owner, ssg_edges, avg_chance, True)

    def solve_value_kleene(self):

        mini = np.iinfo(self.edges.dtype).min

        cur = np.zeros(len(self.owner)).reshape((-1,1))

        while True:
            old = np.array(cur)

            edges_weight = np.where(self.edges != mini, ((1-self.discount)*self.edges)+self.discount*cur, np.nan)

            cur = np.where(self.owner, np.nanmin(edges_weight, 1), np.nanmax(edges_weight, 1))

            # print(edges_weight)

            max_err = np.amax(np.abs(cur-old))

            # iterate until max float precision is hit
            if max_err <1e-14:
                break
        return cur

    def solve_strat_iter(self, player):

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

    def solve(self, strat=None):

        if type(strat) != type(None):

            mini = np.iinfo(self.edges.dtype).min

            old = np.array(self.edges)

            self.edges = np.where(strat!=-1, mini, self.edges.transpose()).transpose()

            for i in np.where(strat!=-1)[0]:
                self.edges[i,strat[i]]=old[i,strat[i]]

            ret = self.solve_value_iter_matrix()

            self.edges = old

        else:

            ret = self.solve_value_iter_matrix()

        return ret

    def solve_strat_value_iter_matrix(self):

        mini = np.iinfo(self.edges.dtype).min

        z = self.solve_value_iter_matrix()

        return np.argmin(np.where(self.edges!=mini, np.abs(((self.discount*z)+((1-self.discount)*self.edges))-z.reshape(-1,1)), 1000),1)

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

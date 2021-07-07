import numpy as np
import os
import pickle
from graphviz import Digraph
from fractions import Fraction
from ortools.linear_solver import pywraplp

import time

class ParityGame:

    def __init__(self, vertices, owner, edges, priority):

        self.vertices = vertices
        self.owner = owner
        self.edges = edges
        self.priority = priority

    def to_mpg(self):

        edges_exist = self.edges
        edges_value = np.fromfunction(lambda x,y: -vertices**priority[x], shape=(vertices,vertices), dtype=int) 

        mini = np.iinfo(edges_value.dtype).min
        edges = np.where(edges_exist, edges_value, mini)

        return MeanPayoffGame(self.vertices, self.owner, edges, 0)

class MeanPayoffGame:

    def __init__(self, vertices, owner, edges, threshold):

        self.vertices = vertices
        self.owner = owner        
        self.edges = edges
        self.threshold = threshold

    def to_dpg(self):

        mini = np.iinfo(self.edges.dtype).min

        empty_as_zero = np.where(self.edges != mini, self.edges, 0)

        W = max(abs(np.amin(empty_as_zero)),abs(np.amax(empty_as_zero)))

        discount = 1-(1/(4*(vertices**3)*W))

        return DiscountedMeanPayoffGame(self.vertices, self.owner, self.edges, self.threshold, discount)

    def to_eg(self):

        return EnergyGame(self.vertices, self.owner, self.edges, 0)

class EnergyGame:

    def __init__(self, vertices, owner, edges, credit):

        self.vertices = vertices
        self.owner = owner        
        self.edges = edges
        self.credit = credit

class DiscountedMeanPayoffGame:

    def __init__(self, vertices, owner, edges, threshold, discount):

        self.vertices = vertices
        self.owner = owner        
        self.edges = edges
        self.threshold = threshold
        self.discount = discount

    def to_ssg(self):

        mini = np.iinfo(self.edges.dtype).min

        empty_as_zero = np.where(self.edges != mini, self.edges, 0)

        W = max(abs(np.amin(empty_as_zero)),abs(np.amax(empty_as_zero)))

        edges = np.where(self.edges != mini, self.edges+W, self.edges)

        f3 = np.count_nonzero(self.edges != mini)

        vertices = self.vertices + f3

        f3pos = np.where(self.edges != mini)

        denominator = np.max(np.where(edges != mini, edges, 0))

        ssg_edges = np.full((vertices,vertices+2), False)

        owner = np.hstack((self.owner, np.full((f3), 2, dtype=np.uint8)))

        avg_chance = np.zeros((f3, vertices+2), dtype=np.uint32)

        for i,edge in enumerate(zip(f3pos[0],f3pos[1])):
            ssg_edges[edge[0], i+self.vertices] = True
            ssg_edges[i+self.vertices, edge[1]] = True
            ssg_edges[i+self.vertices, -1] = True
            ssg_edges[i+self.vertices, -2] = True
            lambda_chance = int(denominator*self.discount/(1-self.discount))
            avg_chance[i, edge[1]] = lambda_chance
            avg_chance[i, -2] = denominator-edges[edge]
            avg_chance[i, -1] = edges[edge]

        return SimpleStochasticGame(vertices, owner, ssg_edges, avg_chance)

    def solve(self):

        mini = np.iinfo(self.edges.dtype).min

        cur = np.zeros(self.vertices).reshape((-1,1))
        for _ in range(100):

            old = np.array(cur)

            edges_weight = np.where(self.edges != mini, ((1-self.discount)*self.edges)+self.discount*cur, np.nan)

            cur = np.where(self.owner, np.nanmin(edges_weight, 1), np.nanmax(edges_weight, 1))

        print([f"{v_n:.3f}" for v_n in cur])

class SimpleStochasticGame:

    def __init__(self, vertices, owner, edges, avg_chance):

        self.vertices = vertices
        self.owner = owner        
        self.edges = edges
        self.avg_chance = avg_chance

if __name__ == '__main__':

    N = 2**5
    p = 0.3125

    p=((p*N)-1)/(N-1)

    # # pg demo
    # vertices = N
    # owner = np.random.choice([False, True], size=(vertices))
    # edges = np.random.choice([False, True], size=(vertices,vertices))
    # priority = np.random.randint(0, vertices, size=(vertices))
    # pg = ParityGame(vertices,owner,edges,priority)
    # print(pg)
    # mpg = pg.to_mpg()
    # print(mpg)

    # # mpg demo
    # vertices = N
    # owner = np.random.choice([False, True], size=(vertices))
    # edges_exist = np.random.choice([False, True], size=(vertices,vertices))
    # edges_value = np.random.randint(-10, 11, size=(vertices,vertices))
    # mini = np.iinfo(edges_value.dtype).min
    # edges = np.where(edges_exist, edges_value, mini)
    # threshold = np.random.randint(-vertices,vertices, size=(1))
    # mpg = MeanPayoffGame(vertices, owner, edges, threshold)
    # print(mpg)
    # dpg = mpg.to_dpg()
    # print(dpg)
    # eg = mpg.to_eg()
    # print(eg)

    # # eg demo
    # vertices = N
    # owner = np.random.choice([False, True], size=(vertices))
    # edges_exist = np.random.choice([False, True], size=(vertices,vertices))
    # edges_value = np.random.randint(-10, 11, size=(vertices,vertices))
    # mini = np.iinfo(edges_value.dtype).min
    # edges = np.where(edges_exist, edges_value, mini)
    # credit = np.random.randint(-vertices*10,vertices*10, size=(1))
    # eg = EnergyGame(vertices, owner, edges, credit)
    # print(eg)

    # dpg demo
    # vertices = N
    # owner = np.random.choice([False, True], size=(vertices))
    # edges_exist = np.random.choice([False, True], size=(vertices,vertices))
    # edges_value = np.random.randint(-10, 11, size=(vertices,vertices))
    # mini = np.iinfo(edges_value.dtype).min
    # edges = np.where(edges_exist, edges_value, mini)
    # threshold = np.random.randint(-vertices,vertices, size=(1))
    # discount = np.random.rand(1)
    # dpg = DiscountedMeanPayoffGame(vertices, owner, edges, threshold, discount)
    # print(dpg)
    # dpg.solve()
    # ssg = dpg.to_ssg()
    # print(ssg)

    # # ssg demo
    # vertices = N
    # owner = np.random.randint(0, 3, size=(vertices), dtype=np.int8)
    # edges = np.random.choice([False, True], size=(vertices,vertices+2))
    # avg_n = len(np.where(owner==2)[0])
    # avg_chance = np.random.randint(0, 256, size=(avg_n,vertices+2), dtype=np.uint8)
    # avg_chance = np.where(edges[np.where(owner==2)], avg_chance, 0)
    # ssg = SimpleStochasticGame(vertices, owner, edges, avg_chance)
    # print(ssg)

    vertices = N
    owner = np.random.choice([False, True], size=(vertices))
    edges_exist = np.zeros((vertices,vertices), dtype=bool)
    for n in edges_exist:
        rng = np.random.randint(vertices)
        n[rng] = True
        n[:rng] = np.random.choice([True, False], size=rng, p=[p, 1-p])
        n[rng+1:] = np.random.choice([True, False], size=vertices-(rng+1), p=[p, 1-p])
    edges_value = np.random.randint(-10, 11, size=(vertices,vertices))
    mini = np.iinfo(edges_value.dtype).min
    edges = np.where(edges_exist, edges_value, mini)
    threshold = np.random.randint(-vertices,vertices, size=(1))
    discount = np.random.rand(1)
    dpg = DiscountedMeanPayoffGame(vertices, owner, edges, threshold, discount)

    # filename = os.path.join(r"C:\ata\uni\master\grest\grest", "test.bin")
    # file = open(filename, "wb")
    # pickle.dump(dpg, file)
    # file.close()

    # filename2 = os.path.join(r"C:\ata\uni\master\grest\grest", "save2inf.bin")
    # file = open(filename2,"rb")
    # dpg = pickle.load(file)
    # file.close()

    # filename = os.path.join(r"C:\ata\uni\master\grest\grest", "save4inf.bin")
    # file = open(filename, "wb")
    # pickle.dump(dpg, file)
    # file.close()
    # exit()

    # view = Digraph(format="png")
    # for i,node in enumerate(dpg.owner):
    #     if not node:
    #         view.node(str(i), shape="square")
    #     else:
    #         view.node(str(i), shape="circle")
    # idx = np.where(dpg.edges != np.iinfo(dpg.edges.dtype).min)
    # for j,_ in enumerate(idx[0]):
    #     view.edge(str(idx[0][j]),str(idx[1][j]),str(dpg.edges[idx[0][j],idx[1][j]]))
    # view.render(filename=os.path.join(r"C:\ata\uni\master\grest\grest", "test"), view=False, cleanup=True)

    ################

    p0 = np.where(dpg.owner==False)[0]
    p1 = np.where(dpg.owner==True)[0]

    mini = np.iinfo(dpg.edges.dtype).min

    rnd_strat = np.apply_along_axis(lambda x: np.random.choice(np.where(x!=mini)[0]), 1, dpg.edges[p0])

    rnd_strat = np.vstack((p0,rnd_strat))

    rnd_strat = np.transpose(rnd_strat)

    strat = rnd_strat

    strat_hist = []

    while not hash(strat.tobytes()) in strat_hist:

        strat_hist.append(hash(strat.tobytes()))

        weights = dpg.edges[np.where(dpg.edges!=mini)]
        W = max(abs(np.amin(weights)),abs(np.amax(weights)))

        solver = pywraplp.Solver.CreateSolver('GLOP')

        v = [solver.NumVar(float(-W), float(W), str(x)) for x in range(dpg.vertices)]

        for s,t in strat:
            solver.Add(v[s] == (1-float(dpg.discount))*float(dpg.edges[s,t])+float(dpg.discount)*v[t])

        for s in p1:
            for t in np.where(dpg.edges[s]!=mini)[0]:
                solver.Add(v[s] <= (1-float(dpg.discount))*float(dpg.edges[s,t])+float(dpg.discount)*v[t])

        obj_func = v[0]

        for v_n in v[1:]:
            obj_func+=v_n

        solver.Maximize(obj_func)

        status = solver.Solve()

        # print([f"{v_n.solution_value():.2f}" for v_n in v])

        strat = np.empty(0, dtype=np.int64)

        for s in p0:
            idx = np.where(dpg.edges[s]!=mini)[0]
            new = np.argmax([(1-dpg.discount)*dpg.edges[s,t]+dpg.discount*v[t].solution_value() for t in idx])
            strat = np.hstack((strat, idx[new]))

        strat = np.transpose(np.vstack((p0,strat)))

    print(strat)

    print([f"{v_n.solution_value():.3f}" for v_n in v])

    print(dpg.solve())

    ################

    # ssg = dpg.to_ssg()

    # view = Digraph(format="png")
    # for i,node in enumerate(ssg.owner):
    #     if node==0:
    #         view.node(str(i), shape="square")
    #     elif node==1:
    #         view.node(str(i), shape="circle")
    #     else:
    #         view.node(str(i), shape="diamond")
    # view.node(str(ssg.vertices), label="MIN", shape="triangle")
    # view.node(str(ssg.vertices+1), label="MAX", shape="triangle")

    # normal = ssg.vertices-ssg.avg_chance.shape[0]

    # idx = np.where(ssg.edges[:normal])
    # for s,t in zip(idx[0],idx[1]):
    #     view.edge(str(s),str(t))

    # idx = np.where(ssg.edges[normal:])
    # for s,t in zip(idx[0],idx[1]):
    #     view.edge(str(normal+s),str(t),f"{(ssg.avg_chance[s,t]/np.sum(ssg.avg_chance[s])):.3f}")

    # view.render(filename=os.path.join(r"C:\ata\uni\master\grest\grest", "test"), view=False, cleanup=True)

    # p0 = np.where(ssg.owner==0)[0]
    # p1 = np.where(ssg.owner==1)[0]
    # p2 = np.where(ssg.owner==2)[0]

    # rnd_strat = np.apply_along_axis(lambda x: np.random.choice(np.where(x)[0]), 1, ssg.edges[p0])

    # rnd_strat = np.vstack((p0,rnd_strat))

    # rnd_strat = np.transpose(rnd_strat)

    # strat = rnd_strat

    # strat_hist = []

    # while not hash(strat.tobytes()) in strat_hist:

    #     strat_hist.append(hash(strat.tobytes()))

    #     solver = pywraplp.Solver.CreateSolver('GLOP')

    #     v = [solver.NumVar(float(0), float(1), str(x)) for x in range(ssg.vertices)]+[solver.NumVar(float(0), float(0), str(ssg.vertices+1))]+[solver.NumVar(float(1), float(1), str(ssg.vertices+2))]

    #     for s,t in strat:
    #         solver.Add(v[s] == v[t])

    #     for s in p1:
    #         for t in np.where(ssg.edges[s])[0]:
    #             solver.Add(v[s] <= v[t])

    #     normal = ssg.vertices-ssg.avg_chance.shape[0]

    #     for s in p2:
    #         where = np.where(ssg.edges[s])[0]
    #         su = np.sum(ssg.avg_chance[s-normal,where])
    #         solver.Add(v[s] == ((ssg.avg_chance[s-normal,where[0]]/su)*v[where[0]]+(ssg.avg_chance[s-normal,where[1]]/su)*v[where[1]]+(ssg.avg_chance[s-normal,where[2]]/su)*v[where[2]]))

    #     obj_func = v[0]

    #     for v_n in v[1:]:
    #         obj_func+=v_n

    #     solver.Maximize(obj_func)

    #     status = solver.Solve()

    #     strat = np.empty(0, dtype=np.int64)

    #     for s in p0:
    #         idx = np.where(ssg.edges[s])[0]
    #         new = np.argmax([v[t].solution_value() for t in idx])
    #         strat = np.hstack((strat, idx[new]))

    #     strat = np.transpose(np.vstack((p0,strat)))

    # print(strat)

    # print([f"{v_n.solution_value():.3f}" for v_n in v])
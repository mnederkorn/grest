import numpy as np
import os
import pickle
import time
from graphviz import Digraph
from fractions import Fraction
from ortools.linear_solver import pywraplp
from datetime import datetime
from itertools import *
import math
import timeit
import functools

class Game:

    def __init__(self, vertices, owner, edges):

        if not isinstance(vertices, int): raise TypeError("not isinstance(vertices, int)")
        if not isinstance(owner, np.ndarray): raise TypeError("not isinstance(owner, np.ndarray)")
        if not isinstance(edges, np.ndarray): raise TypeError("not isinstance(edges, np.ndarray)")
        if not (vertices,) == owner.shape: raise TypeError("not (vertices,) == owner.shape")

        self.vertices = vertices
        self.owner = owner
        self.edges = edges

    def save(self, target_path=None):

        if target_path == None:
            target_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "graphs", f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.bin")

        with open(target_path, "wb") as file:
            pickle.dump(self, file)

        return target_path

class ParityGame(Game):

    def __init__(self, vertices, owner, edges, priority):

        if not owner.dtype == bool: raise TypeError("not owner.dtype == bool")
        if not edges.dtype == bool: raise TypeError("not edges.dtype == bool")
        if not isinstance(priority, np.ndarray): raise TypeError("not isinstance(priority, np.ndarray)")
        if not priority.dtype == int: raise TypeError("not priority.dtype == int")
        if not (edges.shape == (vertices,vertices)): raise TypeError("not (edges.shape == (vertices,vertices))")
        if not (vertices,) == priority.shape: raise TypeError("not (vertices,) == priority.shape")

        super().__init__(vertices, owner, edges)
        self.priority = priority

    @classmethod
    def generate(cls, n, p):
        vertices = n
        owner = np.random.choice([False, True], size=(vertices))
        edges = np.empty((vertices,vertices), dtype=bool)
        for n in edges:
            rng = np.random.randint(vertices)
            n[rng] = True
            n[:rng] = np.random.choice([False, True], size=rng, p=[1-p, p])
            n[rng+1:] = np.random.choice([False, True], size=vertices-(rng+1), p=[1-p, p])
        priority = np.random.randint(0, vertices+1, size=(vertices))

        return cls(vertices, owner, edges, priority)

    def to_mpg(self):

        edges_exist = self.edges
        edges_value = np.fromfunction(lambda x,y: -self.vertices**self.priority[x], shape=(self.vertices,self.vertices), dtype=int)

        mini = np.iinfo(edges_value.dtype).min
        edges = np.where(edges_exist, edges_value, mini)

        return MeanPayoffGame(self.vertices, self.owner, edges, np.array((0,)))

    def visualise(self, target_path=None):

        if target_path == None:
            target_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images", f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}")

        view = Digraph(format="png")
        for i,(owner, priority) in enumerate(zip(self.owner, self.priority)):
            if not owner:
                view.node(str(i), str(priority), shape="square", fontcolor="#000000")
            else:
                view.node(str(i), str(priority), shape="circle", fontcolor="#000000")
        idx = np.where(self.edges==True)
        for s,t in zip(idx[0],idx[1]):
            view.edge(str(s),str(t))
        view.render(filename=target_path, view=False, cleanup=True)

        return target_path

class MeanPayoffGame(Game):

    def __init__(self, vertices, owner, edges, threshold):

        if not owner.dtype == bool: raise TypeError("not owner.dtype == bool")
        if not edges.dtype == np.int32: raise TypeError("not edges.dtype == np.int32")
        if not isinstance(threshold, np.ndarray): raise TypeError("not isinstance(threshold, np.ndarray)")
        if not threshold.dtype == int: raise TypeError("not threshold.dtype == int")
        if not edges.shape == (vertices,vertices): raise TypeError("not edges.shape == (vertices,vertices)")
        if not threshold.shape == (1,): raise TypeError("not threshold.shape == (1,)")

        super().__init__(vertices, owner, edges)
        self.threshold = threshold

    @classmethod
    def generate(cls, n, p, w):
        vertices = n
        owner = np.random.choice([False, True], size=(vertices))
        edges_exist = np.empty((vertices,vertices), dtype=bool)
        for n in edges_exist:
            rng = np.random.randint(vertices)
            n[rng] = True
            n[:rng] = np.random.choice([False, True], size=rng, p=[1-p, p])
            n[rng+1:] = np.random.choice([False, True], size=vertices-(rng+1), p=[1-p, p])
        edges_value = np.random.randint(-w, w+1, size=(vertices,vertices))
        mini = np.iinfo(edges_value.dtype).min
        edges = np.where(edges_exist, edges_value, mini)
        threshold = np.random.randint(-w*n,(w*n)+1, size=((1,)))

        return cls(vertices, owner, edges, threshold)

    def solve_zwick_paterson(self):

        mini = np.iinfo(self.edges.dtype).min

        empty_as_zero = np.where(self.edges != mini, self.edges, 0)

        W = max(abs(np.amin(empty_as_zero)),abs(np.amax(empty_as_zero)))

        a = np.zeros(self.vertices).reshape((-1,1))

        k = (4*(self.vertices**3)*W)

        for x in range(k):

            edges_weight = np.where(self.edges != mini, self.edges+a, np.nan)

            a = np.where(self.owner, np.nanmin(edges_weight, 1), np.nanmax(edges_weight, 1))

        lower = (a/k)-(1/(2*self.vertices*(self.vertices-1)))
        upper = (a/k)+(1/(2*self.vertices*(self.vertices-1)))

        v = np.zeros((self.vertices),dtype=object)

        for v_n in range(self.vertices):
            br=False
            for denominator in range(1,self.vertices+1):
                if br: break
                f=math.floor(lower[v_n]*denominator)
                c=math.ceil(upper[v_n]*denominator)
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

        discount = 1-(1/(4*(self.vertices**3)*W))

        return DiscountedMeanPayoffGame(self.vertices, self.owner, self.edges, self.threshold, np.array((discount,)))

    def to_eg(self):

        return EnergyGame(self.vertices, self.owner, self.edges, np.array((0,)))

    def visualise(self, target_path=None):

        if target_path == None:
            target_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images", f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}")

        view = Digraph(format="png")
        for i,owner in enumerate(self.owner):
            if not owner:
                view.node(str(i), shape="square", fontcolor="#dfdfdf")
            else:
                view.node(str(i), shape="circle", fontcolor="#dfdfdf")
        mini = np.iinfo(self.edges.dtype).min
        idx = np.where(self.edges!=mini)
        for s,t in zip(idx[0],idx[1]):
            view.edge(str(s),str(t),str(self.edges[s,t]))
        view.render(filename=target_path, view=False, cleanup=True)

        return target_path

class EnergyGame(Game):

    def __init__(self, vertices, owner, edges, credit):

        if not owner.dtype == bool: raise TypeError("not owner.dtype == bool")
        if not edges.dtype == np.int32: raise TypeError("not edges.dtype == np.int32")
        if not isinstance(credit, np.ndarray): raise TypeError("not isinstance(credit, np.ndarray)")
        if not credit.dtype == int: raise TypeError("not credit.dtype == int")
        if not edges.shape == (vertices,vertices): raise TypeError("not edges.shape == (vertices,vertices)")
        if not credit.shape == (1,): raise TypeError("not credit.shape == (1,)")

        super().__init__(vertices, owner, edges)
        self.credit = credit

    @classmethod
    def generate(cls, n, p, w):
        vertices = n
        owner = np.random.choice([False, True], size=(vertices))
        edges_exist = np.empty((vertices,vertices), dtype=bool)
        for n in edges_exist:
            rng = np.random.randint(vertices)
            n[rng] = True
            n[:rng] = np.random.choice([False, True], size=rng, p=[1-p, p])
            n[rng+1:] = np.random.choice([False, True], size=vertices-(rng+1), p=[1-p, p])
        edges_value = np.random.randint(-w, w+1, size=(vertices,vertices))
        mini = np.iinfo(edges_value.dtype).min
        edges = np.where(edges_exist, edges_value, mini)
        credit = np.random.randint(-w*n,(w*n)+1, size=((1,)))

        return cls(vertices, owner, edges, credit)

    def solve_bcdgr(self):

        p0 = np.where(self.owner==False)[0]
        p1 = np.where(self.owner==True)[0]

        mini = np.iinfo(self.edges.dtype).min

        l = set()

        # M_(G^gamma)
        max_cycle_cost = 0
        for v in self.edges:
            max_cycle_cost += np.max((0,np.max(-(v[v!=mini]))))

        def minus(a,b):
            if (a!=-1 and ((aminb:=(a-b))<=max_cycle_cost)):
                return max(0,aminb)
            else:
                return -1

        def leq(x,y):
            if (y==-1 or 0<=x<=y<=max_cycle_cost):
                return True
            else:
                return False

        for v in p0:
            if np.all(self.edges[v]<0):
                l.add(v)

        for v in p1:
            if np.any(np.logical_and(self.edges[v]<0,self.edges[v]!=mini)):
                l.add(v)

        f = np.zeros(self.vertices)

        cnt = np.zeros(self.vertices)

        for v in p0:
            for w in np.where(self.edges[v]!=mini)[0]:
                if leq(minus(f[w],self.edges[v,w]),f[v]):
                    cnt[v]+=1

        while l:
            v=l.pop()
            old=f[v]
            if not self.owner[v]:
                f[v]=min([minus_w for w in np.where(self.edges[v]!=mini)[0] if (minus_w:=minus(f[w],self.edges[v,w]))!=-1], default=-1)
            else:
                ma=0
                cand=None
                for w in np.where(self.edges[v]!=mini)[0]:
                    minus_w=minus(f[w],self.edges[v,w])
                    if minus_w==-1:
                        ma=-1
                        cand=w
                        break
                    elif minus_w>ma:
                        cand=w
                        ma=minus_w
                f[v]=ma
            if not self.owner[v]:
                cnt[v]=0
                for w in np.where(self.edges[v]!=mini)[0]:
                    if leq(minus(f[w],self.edges[v,w]),f[v]):
                        cnt[v]+=1

            for u in [u for u in np.where(self.edges[:,v]!=mini)[0] if not leq(minus(f[v],self.edges[u,v]),f[u])]:
                if not self.owner[u]:
                    if leq(minus(old,self.edges[u,v]),f[u]):
                        cnt[u]-=1
                    if cnt[u]<=0:
                        l.add(u)
                else:
                    l.add(u)

        return f

    def solve_value_iter(self):

        p0 = np.where(self.owner==False)[0]
        p1 = np.where(self.owner==True)[0]

        mini = np.iinfo(self.edges.dtype).min

        l = set()

        # M_(G^gamma)
        max_cycle_cost = 0
        for v in self.edges:
            max_cycle_cost += np.max((0,np.max(-(v[v!=mini]))))

        def minus(a,b):
            if (a!=-1 and ((aminb:=(a-b))<=max_cycle_cost)):
                return max(0,aminb)
            else:
                return -1

        vminus = np.vectorize(minus)

        def leq(x,y):
            if (y==-1 or 0<=x<=y<=max_cycle_cost):
                return True
            else:
                return False

        vleq = np.vectorize(leq)

        f = np.zeros(self.vertices).reshape((-1,1))

        while True:

            print(max_cycle_cost, f)

            old = np.array(f)

            # edges_weight = np.where(self.edges != mini, ((1-self.discount)*self.edges)+self.discount*cur, np.nan)
            # cur = np.where(self.owner, np.nanmin(edges_weight, 1), np.nanmax(edges_weight, 1))

            edges_weight = np.where(self.edges != mini, vminus(f,self.edges), np.nan)

            f = np.where(self.owner, np.nanmin(edges_weight, 1), np.nanmax(edges_weight, 1))

            # f[v]=min([minus_w for w in np.where(self.edges[v]!=mini)[0] if (minus_w:=minus(f[w],self.edges[v,w]))!=-1], default=-1)
            # else:
            #     ma=0
            #     for w in np.where(self.edges[v]!=mini)[0]:
            #         minus_w=minus(f[w],self.edges[v,w])
            #         if minus_w==-1:
            #             ma=-1
            #             break
            #         elif minus_w>ma:
            #             ma=minus_w
            #     f[v]=ma

            if np.all(f==old):
                break

        return f

    def visualise(self, target_path=None):

        if target_path == None:
            target_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images", f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}")

        view = Digraph(format="png")
        for i,owner in enumerate(self.owner):
            if not owner:
                view.node(str(i), shape="square", fontcolor="#000000")
            else:
                view.node(str(i), shape="circle", fontcolor="#000000")
        mini = np.iinfo(self.edges.dtype).min
        idx = np.where(self.edges!=mini)
        for s,t in zip(idx[0],idx[1]):
            view.edge(str(s),str(t),str(self.edges[s,t]))
        view.render(filename=target_path, view=False, cleanup=True)

        return target_path

class DiscountedMeanPayoffGame(Game):

    def __init__(self, vertices, owner, edges, threshold, discount):

        if not owner.dtype == bool: raise TypeError("not owner.dtype == bool")
        if not edges.dtype == np.int32: raise TypeError("not edges.dtype == np.int32")
        if not isinstance(threshold, np.ndarray): raise TypeError("not isinstance(threshold, np.ndarray)")
        if not isinstance(discount, np.ndarray): raise TypeError("not isinstance(discount, np.ndarray)")
        if not threshold.dtype == int: raise TypeError("not threshold.dtype == int")
        if not discount.dtype == float: raise TypeError("not discount.dtype == float")
        if not edges.shape == (vertices,vertices): raise TypeError("not edges.shape == (vertices,vertices)")
        if not threshold.shape == (1,): raise TypeError("not threshold.shape == (1,)")
        if not discount.shape == (1,): raise TypeError("not discount.shape == (1,)")

        super().__init__(vertices, owner, edges)
        self.threshold = threshold
        self.discount = discount

    @classmethod
    def generate(cls, n, p, w):
        vertices = n
        owner = np.random.choice([False, True], size=(vertices))
        edges_exist = np.empty((vertices,vertices), dtype=bool)
        for n in edges_exist:
            rng = np.random.randint(vertices)
            n[rng] = True
            n[:rng] = np.random.choice([False, True], size=rng, p=[1-p, p])
            n[rng+1:] = np.random.choice([False, True], size=vertices-(rng+1), p=[1-p, p])
        edges_value = np.random.randint(-w, w+1, size=(vertices,vertices))
        mini = np.iinfo(edges_value.dtype).min
        edges = np.where(edges_exist, edges_value, mini)
        threshold = np.random.randint(-w*n,(w*n)+1, size=((1,)))
        discount = np.random.rand(1)

        return cls(vertices, owner, edges, threshold, discount)

    def to_ssg(self):

        mini = np.iinfo(self.edges.dtype).min

        empty_as_zero = np.where(self.edges != mini, self.edges, 0)

        W = max(abs(np.amin(empty_as_zero)),abs(np.amax(empty_as_zero)))

        edges = np.where(self.edges != mini, self.edges+W, self.edges)

        f3 = np.count_nonzero(self.edges != mini)

        vertices = self.vertices + f3

        f3pos = np.where(self.edges != mini)

        denominator = np.iinfo(np.uint32).max

        ssg_edges = np.full((vertices,vertices+2), False)

        owner = np.hstack((self.owner, np.full(f3, 2, dtype=np.uint8)))

        avg_chance = np.zeros((f3, vertices+2), dtype=np.uint32)

        for i,edge in enumerate(zip(f3pos[0],f3pos[1])):
            ssg_edges[edge[0], i+self.vertices] = True
            ssg_edges[i+self.vertices, edge[1]] = True
            ssg_edges[i+self.vertices, -1] = True
            ssg_edges[i+self.vertices, -2] = True
            avg_chance[i, edge[1]] = int(denominator*self.discount)
            avg_chance[i, -2] = int(denominator*(1-self.discount)*(1-(edges[edge]/(2*W))))
            avg_chance[i, -1] = int(denominator*(1-self.discount)*(edges[edge]/(2*W)))

        return SimpleStochasticGame(vertices, owner, ssg_edges, avg_chance), W

    def solve_value_iter(self):

        mini = np.iinfo(self.edges.dtype).min

        cur = np.zeros(self.vertices).reshape((-1,1))

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

            v = [solver.NumVar(float(-W), float(W), str(x)) for x in range(self.vertices)]

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

        # print(strat)

        return np.array([v_n.solution_value() for v_n in v])

    def visualise(self, target_path=None):

        if target_path == None:
            target_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images", f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}")

        view = Digraph(format="png")
        for i,owner in enumerate(self.owner):
            if not owner:
                view.node(str(i), shape="square", fontcolor="#dfdfdf")
            else:
                view.node(str(i), shape="circle", fontcolor="#dfdfdf")
        mini = np.iinfo(self.edges.dtype).min
        idx = np.where(self.edges!=mini)
        for s,t in zip(idx[0],idx[1]):
            view.edge(str(s),str(t),str(self.edges[s,t]))
        view.render(filename=target_path, view=False, cleanup=True)

        return target_path

class SimpleStochasticGame(Game):

    def __init__(self, vertices, owner, edges, avg_chance):

        if not owner.dtype == np.uint8: raise TypeError("not owner.dtype == np.uint8")
        if not edges.dtype == bool: raise TypeError("not edges.dtype == bool")
        if not isinstance(avg_chance, np.ndarray): raise TypeError("not isinstance(avg_chance, np.ndarray)")
        if not avg_chance.dtype == np.uint32: raise TypeError("not avg_chance.dtype == np.uint32")
        if not avg_chance.shape[1:] == (vertices+2,): raise TypeError("not avg_chance.shape[1:] == (vertices+2,)")

        super().__init__(vertices, owner, edges)
        self.avg_chance = avg_chance

    @classmethod
    def generate(cls, n, p):
        vertices = n
        owner = np.random.randint(0, 3, size=(vertices), dtype=np.uint8)
        edges = np.empty((vertices,vertices+2), dtype=bool)
        for n in edges:
            rng = np.random.randint(vertices+2)
            n[rng] = True
            n[:rng] = np.random.choice([False, True], size=rng, p=[1-p, p])
            n[rng+1:] = np.random.choice([False, True], size=(vertices+2)-(rng+1), p=[1-p, p])

        avg_n = np.count_nonzero(owner==2)
        avg_chance = np.random.randint(0, 256, size=(avg_n,vertices+2), dtype=np.uint32)
        avg_chance = np.where(edges[np.where(owner==2)], avg_chance, 0)

        return cls(vertices, owner, edges, avg_chance)

    def solve_value_iter(self):

        cur = np.hstack((np.zeros(self.vertices), [0], [1]))

        while True:

            old = np.array(cur)

            edges_weight = np.tile(cur, (self.vertices,1))
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

        p0 = np.where(ssg.owner==0)[0]
        p1 = np.where(ssg.owner==1)[0]
        p2 = np.where(ssg.owner==2)[0]

        rnd_strat = np.apply_along_axis(lambda x: np.random.choice(np.where(x)[0]), 1, ssg.edges[p0])

        rnd_strat = np.vstack((p0,rnd_strat))

        rnd_strat = np.transpose(rnd_strat)

        strat = rnd_strat

        strat_hist = []

        while not hash(strat.tobytes()) in strat_hist:

            strat_hist.append(hash(strat.tobytes()))

            solver = pywraplp.Solver.CreateSolver('GLOP')

            v = [solver.NumVar(float(0), float(1), str(x)) for x in range(ssg.vertices)]+[solver.NumVar(float(0), float(0), str(ssg.vertices+1))]+[solver.NumVar(float(1), float(1), str(ssg.vertices+2))]

            for s,t in strat:
                solver.Add(v[s] == v[t])

            for s in p1:
                for t in np.where(ssg.edges[s])[0]:
                    solver.Add(v[s] <= v[t])

            for i,s in enumerate(p2):
                where = np.where(ssg.edges[s])[0]
                su = np.sum(ssg.avg_chance[i,where])
                val = sum([v[val_n]*(ssg.avg_chance[i,val_n]/su) for val_n in where])
                solver.Add(v[s] == val)

            obj_func = v[0]

            for v_n in v[1:]:
                obj_func+=v_n

            solver.Maximize(obj_func)

            status = solver.Solve()

            strat = np.empty(0, dtype=np.int64)

            for s in p0:
                idx = np.where(ssg.edges[s])[0]
                new = np.argmax([v[t].solution_value() for t in idx])
                strat = np.hstack((strat, idx[new]))

            strat = np.transpose(np.vstack((p0,strat)))

        # print(strat)

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
        view.node(str(self.vertices), label="MIN", shape="diamond", peripheries="2")
        view.node(str(self.vertices+1), label="MAX", shape="diamond", peripheries="2")

        for s in np.where(self.owner!=2)[0]:
            for t in np.where(self.edges[s]==True)[0]:
                view.edge(str(s),str(t))

        for i,s in enumerate(np.where(self.owner==2)[0]):
            total = np.sum(self.avg_chance[i])
            for t in np.where(self.edges[s]==True)[0]:
                view.edge(str(s),str(t),f"{self.avg_chance[i,t]}/{total}")

        view.render(filename=target_path, view=False, cleanup=True)

        return target_path

def load(target_path):
    if os.path.isfile(target_path):
        with open(target_path, "rb") as file:
            try:
                game = pickle.load(file)
                return game
            except Exception as e:
                print(e)

if __name__ == '__main__':

    n = int((2**.5)**8)

    # outdegree of every vertex has to be >=1
    # p is taken as if this wasn't the case
    # to adjust, every vertex is given at least one outgoing edge and p is rescaled afterwards
    # expected value of outdegree of vertices stays the same but the distribution changes
    # to allow for rescaling while keeping expected value of outdegrees, p has to be 1/#V<=p<=1
    p = 2/n
    p=((p*n)-1)/(n-1)

    w=10

    # # pg demo
    # pg = ParityGame.generate(n, p)
    # pg.visualise()
    # mpg = pg.to_mpg()
    # mpg.visualise()

    # mpg demo
    # mpg = MeanPayoffGame.generate(n, p, w)
    # mpg.visualise()
    # mpg_solve_v = mpg.solve_zwick_paterson()
    # print(mpg_solve_v)


    # # eg demo
    # eg = load(r"C:\ata\uni\master\grest\grest\graphs\EnergyGame_2021-08-24-08-54-53-852651.bin"))
    eg = EnergyGame.generate(n, p, w)
    # eg.visualise()
    # eg.save()
    eg_solve_v = eg.solve_bcdgr()
    eg_value_v = eg.solve_value_iter()

    print("------")
    print(eg_value_v)
    print(eg_solve_v)

    # # dpg demo
    # dpg = DiscountedMeanPayoffGame.generate(n, p, w)
    # dpg = mpg.to_dpg()
    # dpg_strat_v = dpg.solve_strat_iter()
    # print([f"{v_n:.3f}" for v_n in dpg_strat_v])
    # dpg_value_v = dpg.solve_value_iter()
    # print([f"{v_n:.3f}" for v_n in dpg_value_v])
    # ssg, W = dpg.to_ssg()

    # ssg demo
    # ssg = SimpleStochasticGame.generate(n, p)
    # ssg_strat_v = ssg.solve_strat_iter()
    # print([f"{v_n:.3f}" for v_n in ssg_strat_v])
    # ssg_strat_v_rescaled = (ssg_strat_v*2*W)-W
    # print([f"{v_n:.3f}" for v_n in ssg_strat_v_rescaled[:dpg.vertices]])
    # ssg_value_v = ssg.solve_value_iter()
    # print([f"{v_n:.3f}" for v_n in ssg_value_v])
    # ssg_value_v_rescaled = (ssg_value_v*2*W)-W
    # print([f"{v_n:.3f}" for v_n in ssg_value_v_rescaled[:dpg.vertices]])
    # ssg.visualise()
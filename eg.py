from game import *
import numpy as np
from tempfile import gettempdir
from graphviz import Digraph

def find_negative_cycle_nodes(edges):

    mini = np.iinfo(edges.dtype).min
    maxi = np.iinfo(edges.dtype).max

    edges_src = np.vstack((edges,np.zeros((1,edges.shape[1]), dtype=edges.dtype)))
    edges_src = np.hstack((edges_src,np.full((edges_src.shape[0],1), mini, dtype=edges.dtype)))

    dist = np.full(edges_src.shape[0], maxi)
    dist[-1] = 0
    pred = np.full(edges_src.shape[0], -1)

    for i in range(1,edges_src.shape[0]+1):

        src_is_pred = np.tile((dist!=maxi).reshape(-1,1),edges_src.shape[0])
        shorter = (new_dist:=(dist.reshape(-1,1)+edges_src))<dist

        maa=np.ma.masked_array(new_dist, ~(valids:=(edges_src!=mini)&src_is_pred))

        exists_shorter = np.any(valids&shorter,axis=0)

        if i!=(edges_src.shape[0]):
            shorter_idx = np.argmin(maa, axis=0)

            pred = np.where(exists_shorter, shorter_idx, pred)
            dist = np.where(exists_shorter, new_dist[shorter_idx,np.arange(edges_src.shape[1])], dist)

        else:

            cycle_nodes_t = set()

            for n in np.where(exists_shorter)[0]:

                cycle_nodes = set()

                s = [n]

                for x in range(edges_src.shape[0]-1):
                    if pred[n] == s[0]:
                        cycle_nodes|=set(s)
                        break
                    else:
                        s.append(pred[n])
                        n=pred[n]

                cycle_nodes_t|=cycle_nodes

            return np.array(list(cycle_nodes_t))

class EnergyGame(Game):

    def __init__(self, owner, edges):

        super().__init__(owner, edges)

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

        return cls(owner, edges)

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

        f = np.zeros(len(self.owner), dtype=int)

        cnt = np.zeros(len(self.owner))

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

        mini = np.iinfo(self.edges.dtype).min
        maxi = np.iinfo(self.edges.dtype).max

        # M_(G^gamma)
        max_cycle_cost = 0
        for v in self.edges:
            max_cycle_cost += np.max((0,np.max(-(v[v!=mini]))))

        def minus(a,b):
            if b==mini:
                return 0
            if (a!=-1 and ((aminb:=(a-b))<=max_cycle_cost)):
                return max(0,aminb)
            else:
                return -1

        vminus = np.vectorize(minus)

        f = np.zeros(len(self.owner),dtype=int).reshape((-1,1))

        while True:

            old = np.array(f)

            edges_weight = vminus(f, self.edges)

            edges_weight =  np.where(self.owner.reshape(-1,1), np.where(self.edges!=mini, edges_weight, mini), np.where(self.edges!=mini, edges_weight, maxi))

            edges_weight = np.where(edges_weight==-1, maxi, edges_weight)

            f = np.where(self.owner, np.max(edges_weight, 1), np.min(edges_weight, 1))

            f = np.where(f==maxi, -1, f)

            if np.all(f==old):
                break

        return f
    
    def solve_strat_iter_below(self):

        p0 = np.where(self.owner==False)[0]
        p1 = np.where(self.owner==True)[0]

        mini = np.iinfo(self.edges.dtype).min
        maxi = np.iinfo(self.edges.dtype).max

        # M_(G^gamma)
        max_cycle_cost = 0
        for v in self.edges:
            max_cycle_cost += np.max((0,np.max(-(v[v!=mini]))))

        nW = len(self.owner)*np.max(np.abs(self.edges[np.where(self.edges!=mini)]))

        edges_p1 = self.edges[np.ix_(p1,p1)]

        cycle_nodes = np.array([],dtype=edges_p1.dtype)

        while True:
            restriction = np.setdiff1d(np.arange(edges_p1.shape[0]),cycle_nodes)
            ret = find_negative_cycle_nodes(edges_p1[np.ix_(restriction,restriction)])
            if ret.shape==(0,):
                break
            else:
                cycle_nodes=np.hstack((cycle_nodes,restriction[ret]))

        if cycle_nodes.shape!=(0,):
            for i in range(1,edges_p1.shape[0]-(cycle_nodes.shape[0]-1)):
                old=cycle_nodes.shape
                cycle_nodes=np.union1d(cycle_nodes,np.nonzero(np.any(edges_p1[:,cycle_nodes]!=mini, axis=1)))
                if cycle_nodes.shape==old:
                    break

        # p1[cycle_nodes] 
        # is the indices of the vertices in V_1 that can reach negative cycles in total control of player 1 (in self.edges indices reference)

        owner = np.hstack((self.owner, False))
        edges = np.vstack((self.edges,np.full((1,self.edges.shape[1]), mini, dtype=self.edges.dtype)))
        edges = np.hstack((edges,np.full((edges.shape[0],1), mini, dtype=self.edges.dtype)))
        edges[-1,-1]=0
        edges[:-1,-1]=np.where(self.owner, edges[:-1,-1], -2*nW)

        # V' & E'
        restriction=np.setdiff1d(np.arange(edges.shape[0]),p1[cycle_nodes])
        owner = owner[restriction]
        edges = edges[np.ix_(restriction,restriction)]

        # strat iteration for guessing for player 1/max player/"energy depleting player"

        p1_ = np.where(owner==True)[0]

        rnd_strat = np.apply_along_axis(lambda v: np.random.choice(np.where(v!=mini)[0]), 1, edges)

        strat = rnd_strat

        strat_hist = []

        while not hash(strat.tobytes()) in strat_hist:

            strat_hist.append(hash(strat.tobytes()))

            f = np.zeros(owner.shape[0],dtype=np.int32)

            oldest = np.array(f)  

            while True:

                old = np.array(f)                

                edges_weight = np.where(edges!=mini,np.maximum(np.minimum(f-edges,3*nW),0),edges)

                edges_weight = np.where(edges_weight==mini, maxi, edges_weight)

                f = np.where(owner, edges_weight[np.arange(edges_weight.shape[0]),strat], np.min(edges_weight, 1))

                edges_weight = np.where(edges_weight==maxi, mini, edges_weight)

                if np.all(f==old):
                    break

            g = edges_weight[np.arange(edges.shape[0]),strat]<edges_weight[np.arange(edges.shape[0]),np.argmax(edges_weight,1)]

            strat = np.where(owner&g, np.argmax(edges_weight,1), strat)

        full = np.full(self.edges.shape[0], -1, dtype=np.int32)

        f = np.where(f<nW, f, -1)[:-1]

        full[np.setdiff1d(np.arange(self.edges.shape[0]), p1[cycle_nodes])]=f

        return full

    def solve_strat_iter_above(self):

        p0 = np.where(self.owner==False)[0]
        p1 = np.where(self.owner==True)[0]

        mini = np.iinfo(self.edges.dtype).min
        maxi = np.iinfo(self.edges.dtype).max

        # M_(G^gamma)
        max_cycle_cost = 0
        for v in self.edges:
            max_cycle_cost += np.max((0,np.max(-(v[v!=mini]))))

        nW = self.vertices*np.max(np.abs(self.edges[np.where(self.edges!=mini)]))

        edges_p1 = self.edges[np.ix_(p1,p1)]

        cycle_nodes = np.array([],dtype=edges_p1.dtype)

        while True:
            restriction = np.setdiff1d(np.arange(edges_p1.shape[0]),cycle_nodes)
            ret = find_negative_cycle_nodes(edges_p1[np.ix_(restriction,restriction)])
            if ret.shape==(0,):
                break
            else:
                cycle_nodes=np.hstack((cycle_nodes,restriction[ret]))

        if cycle_nodes.shape!=(0,):
            for i in range(1,edges_p1.shape[0]-(cycle_nodes.shape[0]-1)):
                old=cycle_nodes.shape
                cycle_nodes=np.union1d(cycle_nodes,np.nonzero(np.any(edges_p1[:,cycle_nodes]!=mini, axis=1)))
                if cycle_nodes.shape==old:
                    break

        # p1[cycle_nodes] 
        # is the indices of the vertices in V_1 that can reach negative cycles in total control of player 1 (in self.edges indices reference)

        owner = np.hstack((self.owner, False))
        edges = np.vstack((self.edges,np.full((1,self.edges.shape[1]), mini, dtype=self.edges.dtype)))
        edges = np.hstack((edges,np.full((edges.shape[0],1), mini, dtype=self.edges.dtype)))
        edges[-1,-1]=0
        edges[:-1,-1]=np.where(self.owner, edges[:-1,-1], -2*nW)

        # V' & E'
        restriction=np.setdiff1d(np.arange(edges.shape[0]),p1[cycle_nodes])
        owner = owner[restriction]
        edges = edges[np.ix_(restriction,restriction)]

        # strat iteration  for player 1/min player/"energy conserving player"

        strat = np.full(edges.shape[0], edges.shape[0]-1, dtype=np.int32)

        strat_hist = np.empty(strat.shape)

        while True:

            strat_hist = np.array(strat)

            solver = pywraplp.Solver.CreateSolver('GLOP')

            v = [solver.NumVar(float(0), float(3*nW), str(x)) for x in range(owner.shape[0])]

            for s,p in enumerate(owner[:-1]):
                if not p:
                    solver.Add(v[s] >= (v[strat[s]]-float(edges[s,strat[s]])))
                else:
                    for t in np.where(edges[s]!=mini)[0]:
                        solver.Add(v[s] >= (v[t]-float(edges[s,t])))

            solver.Add(v[-1] == float(0))

            obj_func = v[0]

            for v_n in v[1:]:
                obj_func+=v_n

            solver.Minimize(obj_func)

            status = solver.Solve()

            v = np.array([v_n.solution_value() for v_n in v])

            while True:

                strat = np.where(owner, strat, np.nanargmin(np.clip(v-np.where(edges!=mini, edges, np.nan), 0, None), 1))

                if np.any(strat!=strat_hist):
                    break

                c1 = (v!=0) # v in [V]^f
                c2 = v.reshape(-1,1)==np.where(edges!=mini, np.clip(v-edges,0,3*nW), mini)
                c3 = 0<v
                c4 = 0<np.clip(v-edges,0,None)
                c5 = np.clip(v-edges,0,None)<=(3*nW)

                V_p0 = c1&np.any(c2&c3&c4&c5, 1)

                V_p1 = c1&np.all((~c2|(c3&c4&c5)), 1)

                V = np.where(owner, V_p1, V_p0)

                if np.any(V) == False:

                    full = np.full(self.edges.shape[0], -1, dtype=np.int32)

                    v = np.where(v<nW, v, -1)[:-1]

                    full[np.setdiff1d(np.arange(self.edges.shape[0]), p1[cycle_nodes])]=v
                    return full
                else:
                    s1 = (v-edges)[np.where((v>edges)&(edges!=mini))]
                    s2 = (v.reshape(-1,1)-v)[np.where(v.reshape(-1,1)>v)]
                    s3 = v[np.where(v!=0)]
                    v = np.where(V, v-np.min([np.min(s1),np.min(s2),np.min(s3)]), v)

    def solve(self, strat=None):

        if type(strat) != type(None):

            mini = np.iinfo(self.edges.dtype).min

            old = np.array(self.edges)

            self.edges = np.where(strat!=-1, mini, self.edges.transpose()).transpose()

            for i in np.where(strat!=-1)[0]:
                self.edges[i,strat[i]]=old[i,strat[i]]

            ret = self.solve_bcdgr()

            self.edges = old

        else:

            ret = self.solve_bcdgr()

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
                label = f"<f(v<sub>{i}</sub>)={float(values[i]):.2f}>" if values[i] != -1 else f"<f(v<sub>{i}</sub>)=&infin;>"
            elif (type(values) == type(None)) and (type(restr_values) != type(None)):
                label = f"<f<sub>|</sub>(v<sub>{i}</sub>)={float(restr_values[i]):.2f}>" if restr_values[i] != -1 else f"<f<sub>|</sub>(v<sub>{i}</sub>)=&infin;>"
            else:
                label = f"<f(v<sub>{i}</sub>)={float(values[i]):.2f}<br/>" if values[i] != -1 else f"<f(v<sub>{i}</sub>)=&infin;<br/>"+f"f<sub>|</sub>(v<sub>{i}</sub>)={float(restr_values[i]):.2f}>" if restr_values[i] != -1 else f"f<sub>|</sub>(v<sub>{i}</sub>)=&infin;>"
            view.node(f"{i}", label=label, shape=shape[owner], fontcolor=colour[owner][strat[i]!=-1], color=colour[owner][strat[i]!=-1])
        mini = np.iinfo(self.edges.dtype).min
        idx = np.where(self.edges!=mini)
        for s,t in zip(idx[0],idx[1]):
            view.edge(str(s),str(t),str(self.edges[s,t]), fontcolor=colour[self.owner[s]][strat[s]==t], color=colour[self.owner[s]][strat[s]==t])

        save_loc = view.render(filename=target_path, view=False, cleanup=True)

        return save_loc

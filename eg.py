from game import *
import numpy as np
from tempfile import gettempdir
from graphviz import Digraph
from numba import jit
from ortools.linear_solver import pywraplp
from itertools import count
import copy
from math import ceil


@jit(nopython=True, cache=True)
def _numba_any_axis0(x):
    out = np.zeros(x.shape[1], dtype=np.bool8)
    for i in range(x.shape[0]):
        out = np.logical_or(out, x[i, :])
    return out


@jit(nopython=True, cache=True)
def _numba_min(x):
    cand = np.where(x != -1)[0]
    if len(cand) != 0:
        y = cand[np.argmin(x[cand])]
        return x[y], y
    else:
        return -1, 0


@jit(nopython=True, cache=True)
def _numba_max(x):
    y = np.where(x == -1)[0]
    if len(y) > 0:
        return x[y[0]], y[0]
    else:
        cand = np.where(x != -1)[0]
        y = cand[np.argmax(x[cand])]
        return x[y], y


# BellmanFord
@jit(nopython=True, cache=True)
def _find_negative_cycle_nodes(edges):

    edges_src = np.vstack((edges, np.zeros((1, edges.shape[1]), dtype=edges.dtype)))
    edges_src = np.hstack(
        (edges_src, np.full((edges_src.shape[0], 1), mini, dtype=edges.dtype))
    )

    dist = np.full(edges_src.shape[0], maxi)
    dist[-1] = 0
    pred = np.full(edges_src.shape[0], -1)

    for i in range(1, edges_src.shape[0] + 1):

        src_is_pred = (dist != maxi).repeat(len(edges_src)).reshape(-1, len(edges_src))
        new_dist = dist.reshape(-1, 1) + edges_src
        shorter = new_dist < dist

        valids = (edges_src != mini) & src_is_pred

        exists_shorter = _numba_any_axis0(valids & shorter)

        shorter_idx = numbafy(np.where(valids, new_dist, maxi), np.argmin, 0)

        pred = np.where(exists_shorter, shorter_idx, pred)
        dist = np.where(exists_shorter, np.diag(new_dist[shorter_idx]), dist)

    cycle_nodes_t = np.full(len(edges), False)

    for n in np.where(exists_shorter)[0]:

        cycle_nodes = np.full(len(edges), False)

        s = [n]

        for x in range(edges_src.shape[0] - 1):
            if pred[n] == s[0]:
                cycle_nodes[np.array(s)] = True
                break
            else:
                s.append(pred[n])
                n = pred[n]

        cycle_nodes_t |= cycle_nodes

    ret = np.where(cycle_nodes_t)[0]

    return pred[ret], ret


# BellmanFord finds (at least) one negative cycle but not necessarily all so we need to iterate
@jit(nopython=True, cache=True)
def _find_all_negative_cycle_nodes(edges):

    neg_strat = np.full(len(edges), -1, dtype=np.int64)

    cycle_nodes = np.full(len(edges), False)

    while True:
        restriction = np.where(~cycle_nodes)[0]
        ret, ret_strat = _find_negative_cycle_nodes(edges[restriction][:, restriction])
        if len(ret) == 0:
            break
        else:
            neg_strat[restriction[ret]] = restriction[ret_strat]
            cycle_nodes[restriction[ret]] = True

    if np.any(cycle_nodes):
        while True:
            old = cycle_nodes.copy()

            adds = numba_any_axis1(edges[:, cycle_nodes] != mini)
            adds[cycle_nodes] = False

            neg_strat[adds] = (np.where(cycle_nodes)[0])[
                numbafy(edges[adds][:, cycle_nodes], np.argmax, 1)
            ]

            cycle_nodes |= adds

            if np.all(cycle_nodes == old):
                break

    return cycle_nodes, neg_strat[cycle_nodes]


class EnergyGame(Game):
    def __init__(self, owner, edges):

        super().__init__(owner, edges)

    @classmethod
    def generate(cls, n, p, w):

        return cls(*super().generate(n, p, w))

    def _solve_both_bcdgr(self, sepm=None):
        @jit(nopython=True, cache=True)
        def _solve_both_bcdgr_jitted(owner, edges, sepm):

            p0 = np.where(owner == False)[0]
            p1 = np.where(owner == True)[0]

            l = np.full(len(owner), False)

            # M_(G^gamma)
            max_cycle_cost = -np.sum(
                numbafy(np.clip(edges, mini, 0) * (edges != mini), np.min, 1)
            )

            def minus(a, b):
                aminb = a - b
                if a != -1 and ((aminb) <= max_cycle_cost):
                    return max(0, aminb)
                else:
                    return -1

            def leq(x, y):
                if y == -1 or 0 <= x <= y <= max_cycle_cost:
                    return True
                else:
                    return False

            for v in p0:
                if np.all(edges[v] < 0):
                    l[v] = True

            for v in p1:
                if np.any(np.logical_and(edges[v] < 0, edges[v] != mini)):
                    l[v] = True

            if sepm == None:
                f = np.full(len(owner), 0, dtype=np.int32)
            else:
                f = sepm

            cnt = np.zeros(len(owner))

            for v in p0:
                for w in np.where(edges[v] != mini)[0]:
                    if leq(minus(f[w], edges[v, w]), f[v]):
                        cnt[v] += 1

            while np.any(l):
                v = np.argmax(l)
                l[v] = False
                old = f[v]
                if not owner[v]:
                    f[v], _ = _numba_min(
                        np.array(
                            [
                                minus(f[w], edges[v, w])
                                for w in np.where(edges[v] != mini)[0]
                            ]
                        )
                    )
                else:
                    f[v], _ = _numba_max(
                        np.array(
                            [
                                minus(f[w], edges[v, w])
                                for w in np.where(edges[v] != mini)[0]
                            ]
                        )
                    )
                if not owner[v]:
                    cnt[v] = 0
                    for w in np.where(edges[v] != mini)[0]:
                        if leq(minus(f[w], edges[v, w]), f[v]):
                            cnt[v] += 1

                for u in [
                    u
                    for u in np.where(edges[:, v] != mini)[0]
                    if not leq(minus(f[v], edges[u, v]), f[u])
                ]:
                    if not owner[u]:
                        if leq(minus(old, edges[u, v]), f[u]):
                            cnt[u] -= 1
                        if cnt[u] <= 0:
                            l[u] = True
                    else:
                        l[u] = True

            return_strat = np.full(len(edges), -1, dtype=np.int32)

            for i in range(len(owner)):
                if not owner[i]:
                    _, cand = _numba_min(
                        np.array(
                            [
                                minus(f[w], edges[i, w])
                                for w in np.where(edges[i] != mini)[0]
                            ]
                        )
                    )
                    return_strat[i] = np.where(edges[i] != mini)[0][cand]

            return f, return_strat

        return _solve_both_bcdgr_jitted(self.owner, self.edges, sepm)

    def _solve_value_fpi(self):

        # M_(G^gamma)
        max_cycle_cost = -np.sum(
            np.min(np.clip(self.edges, mini, 0) * (self.edges != mini), 1)
        )

        f = np.zeros(len(self.owner), dtype=int)

        while True:

            old = np.array(f)

            edges = f - self.edges

            edges_weight = np.where(
                (f != -1) & (edges <= max_cycle_cost), np.clip(edges, 0, maxi), maxi
            )

            edges_weight = np.where(
                self.owner.reshape(-1, 1),
                np.where(self.edges != mini, edges_weight, mini),
                np.where(self.edges != mini, edges_weight, maxi),
            )

            edges_weight = np.where(
                self.owner, np.max(edges_weight, 1), np.min(edges_weight, 1)
            )

            f = np.where(edges_weight == maxi, -1, edges_weight)

            if np.all(f == old):
                return f

    def _solve_strat_fpi(self):

        z = self._solve_value_fpi()

        ret = np.full(len(self.owner), -1, dtype=int)
        edges = np.array(self.edges)

        for i, v in enumerate(edges):
            w = np.where(v != mini)[0]
            while True:
                cl = ceil(len(w) / 2)
                one, two = w[:cl], w[cl:]
                e = edges.copy()
                e[i] = mini
                e[i, one] = edges[i, one]
                x = EnergyGame(self.owner, e)._solve_value_fpi()
                if np.all(x == z):
                    if len(one) == 1:
                        ret[i] = one[0]
                        break
                    else:
                        w = one
                else:
                    w = two
            tmp = edges[i, ret[i]]
            edges[i] = mini
            edges[i, ret[i]] = tmp
        return ret

    def _solve_both_strat_iter_below(self):

        p1 = np.where(self.owner == True)[0]

        nW = len(self.owner) * np.max(np.abs(self.edges[np.where(self.edges != mini)]))

        edges_p1 = self.edges[np.ix_(p1, p1)]

        cycle_nodes, neg_strat = _find_all_negative_cycle_nodes(edges_p1)

        # p1[cycle_nodes]
        # is the indices of the vertices in V_1 that can reach negative cycles in total control of player 1 (in self.edges indices reference)

        owner = np.hstack((self.owner, False))
        edges = np.vstack(
            (
                self.edges,
                np.full((1, self.edges.shape[1]), mini, dtype=self.edges.dtype),
            )
        )
        edges = np.hstack(
            (edges, np.full((edges.shape[0], 1), mini, dtype=self.edges.dtype))
        )
        edges[-1, -1] = 0
        edges[:-1, -1] = np.where(self.owner, edges[:-1, -1], -2 * nW)

        # V' & E'
        restriction = np.setdiff1d(np.arange(len(edges)), p1[cycle_nodes])
        owner = owner[restriction]
        edges = edges[np.ix_(restriction, restriction)]

        # strat iteration for guessing for player 1/"depleting"

        strat = np.where(
            owner,
            np.apply_along_axis(
                lambda v: np.random.choice(np.where(v != mini)[0]), 1, edges
            ),
            -1,
        )

        while True:

            strat_hist = strat.copy()

            f = np.zeros(len(owner), dtype=int)

            while True:

                old = np.array(f)

                edges_weight = np.where(
                    edges != mini, np.maximum(np.minimum(f - edges, 3 * nW), 0), edges
                )

                edges_weight = np.where(edges_weight == mini, maxi, edges_weight)

                f = np.where(
                    owner,
                    edges_weight[np.arange(edges_weight.shape[0]), strat],
                    np.min(edges_weight, 1),
                )

                edges_weight = np.where(edges_weight == maxi, mini, edges_weight)

                if np.all(f == old):
                    break

            g = (
                edges_weight[np.arange(edges.shape[0]), strat]
                < edges_weight[np.arange(edges.shape[0]), np.argmax(edges_weight, 1)]
            )

            strat = np.where(owner & g, np.argmax(edges_weight, 1), strat)

            if np.all(strat_hist == strat):
                break

        full = np.full(self.edges.shape[0], -1, dtype=np.int32)

        f = np.where(f < nW, f, -1)[:-1]

        full[np.setdiff1d(np.arange(self.edges.shape[0]), p1[cycle_nodes])] = f

        return_strat = np.full(len(self.edges), -1)

        if len(cycle_nodes) != 0:
            return_strat[p1[cycle_nodes]] = p1[neg_strat]

        return_strat[
            np.setdiff1d(np.arange(len(self.edges)), p1[cycle_nodes])
        ] = np.where(strat[:-1] != -1, restriction[strat[:-1]], -1)

        return full, return_strat

    def _solve_both_strat_iter_above(self):

        p0 = np.where(self.owner == False)[0]
        p1 = np.where(self.owner == True)[0]

        nW = np.max(
            (
                len(self.owner)
                * np.max(np.abs(self.edges[np.where(self.edges != mini)])),
                1,
            )
        )

        edges_p1 = self.edges[np.ix_(p1, p1)]

        cycle_nodes, neg_strat = _find_all_negative_cycle_nodes(edges_p1)

        # p1[cycle_nodes]
        # is the indices of the vertices in V_1 that can reach negative cycles in total control of player 1 (in self.edges indices reference)

        owner = np.hstack((self.owner, False))
        edges = np.vstack(
            (
                self.edges,
                np.full((1, self.edges.shape[1]), mini, dtype=self.edges.dtype),
            )
        )
        edges = np.hstack(
            (edges, np.full((edges.shape[0], 1), mini, dtype=self.edges.dtype))
        )
        edges[-1, -1] = 0
        edges[:-1, -1] = np.where(self.owner, edges[:-1, -1], -2 * nW)

        # V' & E'
        restriction = np.setdiff1d(np.arange(edges.shape[0]), p1[cycle_nodes])
        owner = owner[restriction]
        edges = edges[np.ix_(restriction, restriction)]

        # strat iteration for player 0/"charging"

        strat = np.where(
            owner, -1, np.full(len(edges), edges.shape[0] - 1, dtype=np.int32)
        )

        while True:

            strat_hist = np.array(strat)

            solver = pywraplp.Solver.CreateSolver("GLOP")

            v = [
                solver.NumVar(float(0), float(3 * nW), str(x))
                for x in range(owner.shape[0])
            ]

            for s, p in enumerate(owner[:-1]):
                if not p:
                    solver.Add(v[s] >= (v[strat[s]] - float(edges[s, strat[s]])))
                else:
                    for t in np.where(edges[s] != mini)[0]:
                        solver.Add(v[s] >= (v[t] - float(edges[s, t])))

            solver.Add(v[-1] == float(0))

            obj_func = v[0]

            for v_n in v[1:]:
                obj_func += v_n

            solver.Minimize(obj_func)

            status = solver.Solve()

            v = np.array([np.int32(np.rint(v_n.solution_value())) for v_n in v])

            while True:

                strat = np.where(
                    owner,
                    strat,
                    np.nanargmin(
                        np.clip(v - np.where(edges != mini, edges, np.nan), 0, None), 1
                    ),
                )

                if np.any(strat != strat_hist):
                    break

                v_ = np.ones(len(v), dtype=bool)

                while True:

                    v_h = v_.copy()

                    c1 = v != 0
                    # c2 also necessitates existance of (v,v')
                    c2 = v.reshape(-1, 1) == np.where(
                        edges != mini, np.clip(v - edges, 0, 3 * nW), mini
                    )
                    c3 = 0 < v
                    c4 = 0 < np.where(edges != mini, np.clip(v - edges, 0, None), mini)
                    c5 = (
                        np.where(edges != mini, np.clip(v - edges, 0, None), mini)
                        <= 3 * nW
                    )
                    c6 = v_

                    V_p0 = c1 & np.any(c2 & c3 & c4 & c5 & c6, 1)

                    V_p1 = c1 & np.all((~c2 | (c3 & c4 & c5 & c6)), 1)

                    V = np.where(owner, V_p1, V_p0)

                    v_ = V

                    if np.all(v_ == v_h):
                        break

                if not np.any(v_):

                    full = np.full(self.edges.shape[0], -1, dtype=np.int32)

                    v = np.where(v < nW, v, -1)[:-1]

                    full[
                        np.setdiff1d(np.arange(self.edges.shape[0]), p1[cycle_nodes])
                    ] = v

                    return_strat = np.full(len(self.edges), -1)
                    if len(cycle_nodes) != 0:
                        return_strat[p1[cycle_nodes]] = p1[neg_strat]

                    return_strat[
                        np.setdiff1d(np.arange(len(self.edges)), p1[cycle_nodes])
                    ] = np.where(strat[:-1] != -1, restriction[strat[:-1]], -1)

                    return_strat = np.where(
                        return_strat == len(self.owner),
                        np.apply_along_axis(
                            lambda v: np.random.choice(np.where(v != mini)[0]),
                            1,
                            self.edges,
                        ),
                        return_strat,
                    )

                    return full, return_strat
                else:
                    v[v_] -= 1

    def solve_value(self, strat=None):

        if type(strat) != type(None):

            edges = np.where((strat == -1).reshape(-1, 1), self.edges, mini)

            for i in np.where(strat != -1)[0]:
                edges[i, strat[i]] = self.edges[i, strat[i]]

            return EnergyGame(self.owner, edges)._solve_both_bcdgr()[0]

        else:

            return self._solve_both_bcdgr()[0]

    def solve_strat(self):

        return self._solve_strat_fpi()

    def solve_both(self):

        return self._solve_both_bcdgr()[0], self._solve_strat_fpi()

    def visualise(self, target_path=None, strat=None, values=None, tmp=False):

        if type(strat) == type(None):
            strat = np.full(len(self.owner), -1)

        if target_path == None:
            if tmp:
                target_path = os.path.join(
                    gettempdir(),
                    f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
                )
            else:
                if target_path == None:
                    target_path = os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "images",
                        f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
                    )

        view = Digraph(format="png")
        view.attr(bgcolor="#f0f0f0")
        for i, owner in enumerate(self.owner):
            if type(values) != type(None):
                label = (
                    f"<f(v<sub>{i}</sub>)={float(values[i]):.2f}>"
                    if values[i] != -1
                    else f"<f(v<sub>{i}</sub>)=&infin;>"
                )
            else:
                label = f"<v<sub>{i}</sub>>"
            view.node(
                f"{i}",
                label=label,
                shape=shape[owner],
                fontcolor=colour[owner][strat[i] != -1],
                color=colour[owner][strat[i] != -1],
            )
        idx = np.where(self.edges != mini)
        for s, t in zip(idx[0], idx[1]):
            view.edge(
                str(s),
                str(t),
                str(self.edges[s, t]),
                fontcolor=colour[self.owner[s]][strat[s] == t],
                color=colour[self.owner[s]][strat[s] == t],
            )

        save_loc = view.render(filename=target_path, view=False, cleanup=True)

        return save_loc

    @staticmethod
    def load_csv(target_path):
        if os.path.isfile(target_path):
            with open(target_path, "r") as file:
                typee = str(file.readline().replace("\n", ""))
                assert typee == "eg"
                owner = file.readline().replace("\n", "")
                owner = owner.split(",")
                owner = np.array(
                    [True if (e == "1" or e == "True") else False for e in owner]
                )
                edges = file.read().split("\n")
                edges = [e.split(",") for e in edges]
                edges = np.array([[int(f) if f else mini for f in e] for e in edges])
            return EnergyGame(owner, edges)

    def save_csv(self, target_path=None):
        if target_path == None:
            target_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "graphs",
                f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv",
            )
        with open(target_path, "w") as file:
            file.write("eg\n")
            file.write(",".join(["1" if e else "0" for e in self.owner]) + "\n")
            file.write(
                "\n".join(
                    [
                        ",".join([str(f) if f != mini else "" for f in e])
                        for e in self.edges
                    ]
                )
            )
        return target_path

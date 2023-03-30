from game import *
import numpy as np
from math import floor, ceil
from dpg import DiscountedPayoffGame
from eg import EnergyGame
from graphviz import Digraph
from tempfile import gettempdir
from numba import jit


def _farey_sequence(n):

    ret = np.zeros((2, 0), dtype=np.int32)

    (a, b, c, d) = (0, 1, 1, n)
    while c <= n:
        k = (n + b) // d
        (a, b, c, d) = (c, d, k * c - a, k * d - b)

        ret = np.append(ret, np.array(((a,), (b,))), 1)

    return ret


@jit(nopython=True, cache=True)
def _trunc(n, v):

    lower = v - (1 / (2 * n * (n - 1)))
    upper = v + (1 / (2 * n * (n - 1)))

    v = np.full(n, 0, dtype=np.float64)

    for v_n in range(n):
        for denominator in range(1, n + 1):
            if floor(lower[v_n] * denominator) != floor(upper[v_n] * denominator):
                v[v_n] = ceil(lower[v_n] * denominator) / denominator
                break
    return v


def _Q_sub_sup(n, mid):

    r = np.arange(1, n + 1)

    l = r * mid

    l1 = np.floor(l).astype(int)
    l2 = np.ceil(l).astype(int)

    ar1 = np.argmax(l1 / r)
    ar2 = np.argmin(l2 / r)

    return np.array([l1[ar1], r[ar1]], dtype=int), np.array(
        [l2[ar2], r[ar2]], dtype=int
    )


def _solve_both_eg_alg(owner, edges, lower, upper):

    a1, a2 = _Q_sub_sup(len(owner), (lower + upper) / 2)

    e1 = np.where(edges != mini, (a1[1] * edges) - a1[0], mini)
    e2 = np.where(edges != mini, (-a1[1] * edges) + a1[0], mini)

    f1, s1 = EnergyGame(owner, e1).solve_both()
    f2, s2 = EnergyGame(~owner, e2).solve_both()

    v = np.empty(len(owner), dtype=float)
    s = np.full(len(owner), -1, dtype=int)

    v = np.where((f1 != -1) & (f2 != -1), a1[0] / a1[1], v)
    s = np.where((f1 != -1) & (f2 != -1), np.where(s1 != -1, s1, s2), s)

    v1 = np.where(f1 == -1)[0]
    v2 = np.where(f2 == -1)[0]

    if len(v1) != 0:
        v[v1], st = _solve_both_eg_alg(
            owner[v1], edges[np.ix_(v1, v1)], lower, a1[0] / a1[1]
        )
        s[v1] = v1[st]
    if len(v2) != 0:
        v[v2], st = _solve_both_eg_alg(
            owner[v2], edges[np.ix_(v2, v2)], a2[0] / a2[1], upper
        )
        s[v2] = v2[st]

    return v, s


class MeanPayoffGame(Game):
    def __init__(self, owner, edges):

        super().__init__(owner, edges)

    @classmethod
    def generate(cls, n, p, w):

        return cls(*super().generate(n, p, w))

    def _solve_value_zwick_paterson(self):
        @jit(nopython=True, cache=True)
        def _solve_value_zwick_paterson_jitted(owner, edges):

            W = np.max(np.abs(np.where(edges != mini, edges, 0)))

            v = np.zeros(len(owner), dtype=np.int64)

            k = 4 * (len(owner) ** 3) * W

            edges = edges.astype(np.int64)

            mini2 = np.iinfo(edges.dtype).min
            maxi2 = np.iinfo(edges.dtype).max

            # this can collide at around 4*(n^3)*(W^2)>2*62 because maxi2-((k*W)+1) can be smaller than v_k and the valid/invalid checks for edges stop making sense; analgous for mini2
            # for W = log_2(|V|) that's around |V|=160000
            # for W = |V|^(1/2) that's around |V|=32768=2^15
            # for W = |V| that's around |V|=4096=2^12

            x = np.where(edges != mini, edges, maxi2 - ((k * W) + 1))
            y = np.where(edges != mini, edges, mini2 + ((k * W) + 1))

            edges = np.empty((len(owner), len(owner)), dtype=edges.dtype)
            for n, (i, j, l) in enumerate(zip(owner, x, y)):
                if i:
                    edges[n] = j
                else:
                    edges[n] = l

            for _ in np.arange(k):

                edges_weight = edges + v

                v = np.where(
                    owner,
                    numbafy(edges_weight, np.min, 1),
                    numbafy(edges_weight, np.max, 1),
                )

            v = v / k

            return _trunc(len(owner), v)

        return _solve_value_zwick_paterson_jitted(self.owner, self.edges)

    def _solve_strat_zwick_paterson(self):

        z = self._solve_value_zwick_paterson()

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
                x = MeanPayoffGame(self.owner, e)._solve_value_zwick_paterson()
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

    # https://doi.org/10.48550/arXiv.1503.04426
    def _solve_both_eg_cr(self):

        v = np.full(len(self.owner), mini)
        strat = np.full(len(self.owner), -1)

        f = np.zeros(len(self.owner))

        W = np.max(np.abs(np.where(self.edges != mini, self.edges, 0)))

        w = np.where(self.edges != mini, self.edges + W, self.edges)

        D = 1

        F_seq = _farey_sequence(len(self.owner))

        for i in range(-W, W + 1):

            F = np.array((0, 1), dtype=np.int32)

            for j in range(1, F_seq.shape[1]):

                prev_f = f.copy()

                prev_w = np.where(w != mini, (1 / D) * w, np.nan)

                prev_F = F.copy()

                F = F_seq[:, j]
                N, D = F

                w = np.where(self.edges != mini, (D * (self.edges - i)) - N, self.edges)

                f = EnergyGame(self.owner, w)._solve_both_bcdgr(
                    np.int32(np.ceil(D * prev_f))
                )[0]

                f = np.where(f != -1, f / D, f)

                v = np.where(
                    ((prev_f != -1) & (f == -1)), i + (prev_F[0] / prev_F[1]), v
                )

        return v

    def _to_dpg(self):

        W = int(
            np.max((np.max(np.abs(np.where(self.edges != mini, self.edges, 0))), 1))
        )

        discount = 1 - (1 / (4 * (len(self.owner) ** 3) * W))

        return DiscountedPayoffGame(self.owner, self.edges, discount)

    def solve_both_dpg(self):

        dpg = self._to_dpg()

        x = dpg.solve_both()

        v, s = x

        return _trunc(len(self.owner), v), s

    def solve_both_eg(self):

        W = np.max(np.abs(np.where(self.edges != mini, self.edges, 0)))

        return _solve_both_eg_alg(self.owner, self.edges, -W, W)

    def solve_value(self, strat=None):

        if type(strat) != type(None):

            edges = np.where((strat == -1).reshape(-1, 1), self.edges, mini)

            for i in np.where(strat != -1)[0]:
                edges[i, strat[i]] = self.edges[i, strat[i]]

            return MeanPayoffGame(self.owner, edges)._solve_value_zwick_paterson()

        else:

            return self._solve_value_zwick_paterson()

    def solve_strat(self):

        return self._solve_strat_zwick_paterson()

    def solve_both(self):

        return self._solve_value_zwick_paterson(), self._solve_strat_zwick_paterson()

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
                label = f"<v(v<sub>{i}</sub>)={float(values[i]):.2f}>"
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
                assert typee == "mpg"
                owner = file.readline().replace("\n", "")
                owner = owner.split(",")
                owner = np.array(
                    [True if (e == "1" or e == "True") else False for e in owner]
                )
                edges = file.read().split("\n")
                edges = [e.split(",") for e in edges]
                edges = np.array([[int(f) if f else mini for f in e] for e in edges])
            return MeanPayoffGame(owner, edges)

    def save_csv(self, target_path=None):
        if target_path == None:
            target_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "graphs",
                f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv",
            )
        with open(target_path, "w") as file:
            file.write("mpg\n")
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

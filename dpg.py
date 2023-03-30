from game import *
import numpy as np
from ssg import SimpleStochasticGame
from graphviz import Digraph
from tempfile import gettempdir
import copy
from numba import jit
from ortools.linear_solver import pywraplp


@jit(nopython=True, cache=True)
def _numba_argmin_axis1(x):
    out = np.empty(x.shape[0], dtype=np.int32)
    for i in range(x.shape[0]):
        out[i] = np.argmin(x[i])
    return out


class DiscountedPayoffGame(Game):
    def __init__(self, owner, edges, discount):

        super().__init__(owner, edges)
        self.discount = discount

    @classmethod
    def generate(cls, n, p, w, d=None):

        if d == None:
            return cls(*super().generate(n, p, w), np.random.rand(1)[0])
        else:
            return cls(*super().generate(n, p, w), d)

    def _solve_both_fpi(self):
        @jit(nopython=True, cache=True)
        def _solve_both_fpi_jitted(owner, edges, discount):

            W = np.float64(np.max(np.abs(np.where(edges != mini, edges, 0))))

            f = np.full(len(owner), -W)

            while True:

                old = f.copy()

                edges_weight = np.where(
                    edges != mini, ((1 - discount) * edges) + discount * f, np.nan
                )

                f = np.where(
                    owner,
                    numbafy(edges_weight, np.nanmin, 1),
                    numbafy(edges_weight, np.nanmax, 1),
                )

                max_err = np.amax(np.abs(f - old))

                if max_err < 1e-14:
                    break

            return f

        value = _solve_both_fpi_jitted(self.owner, self.edges, self.discount)

        strat = np.where(
            self.owner,
            np.argmin(
                np.where(
                    self.edges != mini,
                    (1 - self.discount) * self.edges + self.discount * value,
                    maxi,
                ),
                1,
            ),
            np.argmax(
                np.where(
                    self.edges != mini,
                    (1 - self.discount) * self.edges + self.discount * value,
                    mini,
                ),
                1,
            ),
        )

        return value, strat

    def _solve_both_strat_iter(self, player):

        if not player:
            strat = np.where(
                self.owner,
                -1,
                np.apply_along_axis(
                    lambda x: np.random.choice(np.where(x != mini)[0]), 1, self.edges
                ),
            )
        else:
            strat = np.where(
                self.owner,
                np.apply_along_axis(
                    lambda x: np.random.choice(np.where(x != mini)[0]), 1, self.edges
                ),
                -1,
            )

        while True:

            strat_hist = strat.copy()

            weights = self.edges[np.where(self.edges != mini)]
            W = max(abs(np.amin(weights)), abs(np.amax(weights)))

            solver = pywraplp.Solver.CreateSolver("GLOP")

            v = [
                solver.NumVar(float(-W), float(W), str(x))
                for x in range(len(self.owner))
            ]

            if not player:
                for s, p in enumerate(self.owner):
                    if not p:
                        solver.Add(
                            v[s]
                            == (1 - float(self.discount))
                            * float(self.edges[s, strat[s]])
                            + float(self.discount) * v[strat[s]]
                        )
                    else:
                        for t in np.where(self.edges[s] != mini)[0]:
                            solver.Add(
                                v[s]
                                <= (1 - float(self.discount)) * float(self.edges[s, t])
                                + float(self.discount) * v[t]
                            )
            else:
                for s, p in enumerate(self.owner):
                    if p:
                        solver.Add(
                            v[s]
                            == (1 - float(self.discount))
                            * float(self.edges[s, strat[s]])
                            + float(self.discount) * v[strat[s]]
                        )
                    else:
                        for t in np.where(self.edges[s] != mini)[0]:
                            solver.Add(
                                v[s]
                                >= (1 - float(self.discount)) * float(self.edges[s, t])
                                + float(self.discount) * v[t]
                            )

            obj_func = v[0]
            for v_n in v[1:]:
                obj_func += v_n
            if not player:
                solver.Maximize(obj_func)
            else:
                solver.Minimize(obj_func)
            status = solver.Solve()

            if status != 0:
                # status 2 seems to happen rarely if the linear solver requires floating point precision that exceeds ieee 754 standard
                raise Exception("Could not solve", status)

            if not player:
                strat = np.where(
                    self.owner,
                    strat,
                    np.argmax(
                        ((1 - self.discount) * self.edges)
                        + (
                            self.discount
                            * (np.array([v_n.solution_value() for v_n in v]))
                        ),
                        1,
                    ),
                )
            else:
                strat = np.where(
                    self.owner,
                    np.argmin(
                        (
                            (1 - self.discount)
                            * np.where(self.edges == mini, maxi, self.edges)
                        )
                        + (
                            self.discount
                            * (np.array([v_n.solution_value() for v_n in v]))
                        ),
                        1,
                    ),
                    strat,
                )

            if np.all(strat_hist == strat):
                break

        return np.array([v_n.solution_value() for v_n in v]), strat

    def _to_ssg(self):

        W = np.max((np.max(np.abs(np.where(self.edges != mini, self.edges, 0))), 1))

        edges = np.where(self.edges != mini, (self.edges + W) / (2 * W), self.edges)

        f3 = np.count_nonzero(self.edges != mini)

        vertices = len(self.owner) + f3

        f3pos = np.where(self.edges != mini)

        ssg_edges = np.full((vertices, vertices + 2), False)

        owner = np.hstack((self.owner, np.full(f3, 2, dtype=np.uint8)))

        avg_chance = np.zeros((f3, vertices + 2))

        strat_map = np.zeros(f3, dtype=int)

        for i, edge in enumerate(zip(f3pos[0], f3pos[1])):
            ssg_edges[edge[0], i + len(self.owner)] = True
            ssg_edges[i + len(self.owner), edge[1]] = True
            strat_map[i] = edge[1]
            ssg_edges[i + len(self.owner), -1] = True
            ssg_edges[i + len(self.owner), -2] = True
            avg_chance[i, edge[1]] = self.discount
            avg_chance[i, -2] = (1 - self.discount) * (1 - (edges[edge]))
            avg_chance[i, -1] = (1 - self.discount) * (edges[edge])

        return SimpleStochasticGame(owner, ssg_edges, avg_chance, True), strat_map, W

    def solve_both_ssg(self):

        ssg, smap, W = self._to_ssg()

        v, s = ssg.solve_both()

        v = (v[: len(self.owner)] * 2 * W) - W

        s = smap[(s - len(self.owner))[: len(self.owner)]]

        return v, s

    def solve_value(self, strat=None):

        if type(strat) != type(None):

            edges = np.where((strat == -1).reshape(-1, 1), self.edges, mini)

            for i in np.where(strat != -1)[0]:
                edges[i, strat[i]] = self.edges[i, strat[i]]

            return DiscountedPayoffGame(
                self.owner, edges, self.discount
            )._solve_both_fpi()[0]

        else:

            return self._solve_both_fpi()[0]

    def solve_strat(self):

        return self._solve_both_fpi()[1]

    def solve_both(self):

        return self._solve_both_fpi()

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
                assert typee == "dpg"
                owner = file.readline().replace("\n", "")
                owner = owner.split(",")
                owner = np.array(
                    [True if (e == "1" or e == "True") else False for e in owner]
                )
                discount = float(file.readline())
                edges = file.read().split("\n")
                edges = [e.split(",") for e in edges]
                edges = np.array([[int(f) if f else mini for f in e] for e in edges])
            return DiscountedPayoffGame(owner, edges, discount)

    def save_csv(self, target_path=None):
        if target_path == None:
            target_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "graphs",
                f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv",
            )
        with open(target_path, "w") as file:
            file.write("dpg\n")
            file.write(",".join(["1" if e else "0" for e in self.owner]) + "\n")
            file.write(str(self.discount) + "\n")
            file.write(
                "\n".join(
                    [
                        ",".join([str(f) if f != mini else "" for f in e])
                        for e in self.edges
                    ]
                )
            )
        return target_path

from game import *
import numpy as np
from tempfile import gettempdir
from graphviz import Digraph
from numba import jit
from ortools.linear_solver import pywraplp


class SimpleStochasticGame(Game):
    def __init__(self, owner, edges, avg, stopping):

        super().__init__(owner, edges)
        self.avg = avg
        self.stopping = stopping

    @classmethod
    def generate(cls, n, p):
        assert (
            p >= 1 / n
        ), "Since |post(v)| needs to be >=1 for every v, p needs to be at least p>=1/n"
        p = max(0, min(((p * n) - 1) / (n - 1), 1))
        owner = np.random.randint(0, 3, size=(n), dtype=np.uint8)
        edges = np.zeros((n, n + 2), dtype=bool)
        edges[np.arange(n), np.random.randint(0, n + 2, n)] = True
        edges[edges == False] = np.random.choice(
            [False, True], size=n * (n + 1), p=[1 - p, p]
        )

        e = np.where(owner == 2)
        avg = np.random.random(size=(len(e[0]), n + 2))
        avg = np.where(edges[e], avg, 0)
        avg /= np.sum(avg, 1).reshape(-1, 1)

        return cls(owner, edges, avg, False)

    # strats for avg/rng vertices as -1
    def _solve_both_fpi(self):
        @jit(nopython=True, cache=True)
        def _solve_both_fpi_jitted(owner, edges, avg_chance):

            f = np.hstack((np.zeros(len(owner)), np.array([0]), np.array([1])))

            while True:

                old = f.copy()

                # edges_weight = np.tile(f, (len(owner),1))
                edges_weight = f.repeat(len(owner)).reshape(-1, len(owner)).transpose()
                edges_weight = np.where(edges, edges_weight, np.nan)
                f[:-2] = np.where(
                    owner == 0, numbafy(edges_weight, np.nanmax, 1), f[:-2]
                )
                f[:-2] = np.where(
                    owner == 1, numbafy(edges_weight, np.nanmin, 1), f[:-2]
                )

                idx = np.where(owner == 2)
                f[idx] = np.sum(avg_chance * old, 1)

                max_err = np.amax(np.abs(f - old))

                # iterate until max float precision is hit
                if max_err < 1e-14:
                    break

            return f

        value = _solve_both_fpi_jitted(self.owner, self.edges, self.avg)

        strat = np.where(
            self.owner == 2,
            -1,
            np.where(
                self.owner == 0,
                np.nanargmax(np.where(self.edges, value, np.nan), 1),
                np.nanargmin(np.where(self.edges, value, np.nan), 1),
            ),
        )

        return value, strat

    def _solve_both_strat_iter(self, player):

        assert (
            self.stopping
        ), "SSG needs to be stopping to be solved with strategy iteration. To ensure SSG is stopping, generate DPG and convert to SSG via DiscountedPayoffGame.to_ssg."

        p2 = np.where(self.owner == 2)[0]

        if not player:
            strat = np.where(
                self.owner == 0,
                np.apply_along_axis(
                    lambda x: np.random.choice(np.where(x)[0]), 1, self.edges
                ),
                -1,
            )
        else:
            strat = np.where(
                self.owner == 1,
                np.apply_along_axis(
                    lambda x: np.random.choice(np.where(x)[0]), 1, self.edges
                ),
                -1,
            )

        while True:

            strat_hist = strat.copy()

            solver = pywraplp.Solver.CreateSolver("GLOP")

            v = (
                [
                    solver.NumVar(float(0), float(1), str(x))
                    for x in range(len(self.owner))
                ]
                + [solver.NumVar(float(0), float(0), str(str(len(self.owner) + 1)))]
                + [solver.NumVar(float(1), float(1), str(str(len(self.owner) + 2)))]
            )

            for s, p in enumerate(self.owner):
                if p == player:
                    solver.Add(v[s] == v[strat[s]])
                else:
                    if p == 0:
                        for t in np.where(self.edges[s])[0]:
                            solver.Add(v[s] >= v[t])
                    elif p == 1:
                        for t in np.where(self.edges[s])[0]:
                            solver.Add(v[s] <= v[t])
                    else:
                        val = 0
                        for t in np.where(self.edges[s])[0]:
                            val += v[t] * self.avg[np.nonzero(s == p2)[0][0], t]
                        solver.Add(v[s] == val)

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
                    self.owner == 0,
                    np.argmax(
                        np.where(
                            self.edges,
                            self.edges
                            * (np.array([v_n.solution_value() for v_n in v])),
                            -1,
                        ),
                        1,
                    ),
                    strat,
                )
            else:
                strat = np.where(
                    self.owner == 1,
                    np.argmin(
                        np.where(
                            self.edges,
                            self.edges
                            * (np.array([v_n.solution_value() for v_n in v])),
                            2,
                        ),
                        1,
                    ),
                    strat,
                )

            if np.all(strat_hist == strat):
                break

        return np.array([v_n.solution_value() for v_n in v]), strat

    def solve_value(self, strat=None):

        if type(strat) != type(None):

            edges = (strat == -1).reshape(-1, 1) * self.edges

            for i in np.where(strat != -1)[0]:
                edges[i, strat[i]] = True

            SimpleStochasticGame(self.owner, edges, self.avg, self.stopping).visualise()

            return SimpleStochasticGame(
                self.owner, edges, self.avg, self.stopping
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

        view.node(
            str(len(self.owner)),
            label="0",
            shape=shape[2],
            fontcolor=colour[2][False],
            color=colour[2][False],
            peripheries="2",
        )
        view.node(
            str(len(self.owner) + 1),
            label="1",
            shape=shape[2],
            fontcolor=colour[2][False],
            color=colour[2][False],
            peripheries="2",
        )

        idx = np.where(self.edges == True)
        for s, t in zip(idx[0], idx[1]):
            if self.owner[s] != 2:
                view.edge(
                    str(s),
                    str(t),
                    fontcolor=colour[self.owner[s]][strat[s] == t],
                    color=colour[self.owner[s]][strat[s] == t],
                )

        for i, s in enumerate(np.where(self.owner == 2)[0]):
            for t in np.where(self.edges[s] == True)[0]:
                view.edge(
                    str(s),
                    str(t),
                    f"{self.avg[i,t]:.2f}",
                    fontcolor=colour[2][False],
                    color=colour[2][False],
                )

        save_loc = view.render(filename=target_path, view=False, cleanup=True)

        return save_loc

    @staticmethod
    def load_csv(target_path):
        if os.path.isfile(target_path):
            with open(target_path, "r") as file:
                typee = str(file.readline().replace("\n", ""))
                assert typee == "ssg"
                owner = file.readline().replace("\n", "")
                owner = owner.split(",")
                owner = np.array([int(e) for e in owner])
                stopping = bool(file.readline().replace("\n", ""))
                edges = [file.readline().replace("\n", "") for n in range(len(owner))]
                edges = [e.split(",") for e in edges]
                edges = np.array([[bool(f) for f in e] for e in edges])
                avg = file.read().split("\n")
                avg = [e.split(",") for e in avg]
                avg = np.array(
                    [[float(f) if f != "" else float(0) for f in e] for e in avg]
                )
            return SimpleStochasticGame(owner, edges, avg, stopping)

    def save_csv(self, target_path=None):
        if target_path == None:
            target_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "graphs",
                f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv",
            )
        with open(target_path, "w") as file:
            file.write("ssg\n")
            file.write(",".join([str(e) for e in self.owner]) + "\n")
            file.write(str(self.stopping) + "\n")
            file.write(
                "\n".join([",".join(["1" if f else "" for f in e]) for e in self.edges])
                + "\n"
            )
            file.write(
                "\n".join(
                    [",".join([str(f) if f != 0 else "" for f in e]) for e in self.avg]
                )
            )
        return target_path

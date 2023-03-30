from game import *
import numpy as np
from mpg import MeanPayoffGame
from tempfile import gettempdir
from graphviz import Digraph
from math import ceil
from numba import jit
import numba

e_o = {0: "Even", 1: "Odd"}


@jit(nopython=True, cache=True)
def _find_attractor(player, owner, edges, Nh):

    if len(Nh) == 0:
        return np.empty(0, dtype=numba.boolean), np.empty(0, dtype=numba.int64)

    us = owner == player
    them = owner != player

    atr = Nh.copy()

    strat = np.zeros(atr.shape[0], dtype=numba.int64)
    strat[:] = -1

    while True:
        old = atr.copy()
        us_add = numba_any_axis1(edges[us][:, atr])
        p_strat = numba_argmax_axis1(edges[us][:, atr])
        add_mask = np.zeros(atr.shape[0], dtype=numba.boolean)
        add_mask[(np.where(us)[0])[us_add]] = 1
        strat_add_mask = np.full(atr.shape[0], -1)
        strat_add_mask[np.where(us)[0]] = (np.where(atr)[0])[p_strat]
        strat = np.where(add_mask & (strat == -1), strat_add_mask, strat)
        them_add = ~numba_any_axis1(edges[them][:, ~atr])
        atr[(np.where(us)[0])[us_add]] = True
        atr[(np.where(them)[0])[them_add]] = True
        if np.all(atr == old):
            return atr, strat


def _zielonka(owner, edges, priorities):

    if len(owner) == 0:
        return np.array([], dtype=bool), np.array([], dtype=int)
    else:
        m = np.max(priorities)
        player = m % 2
        A, s = _find_attractor(player, owner, edges, (priorities == m))
        z1, se = _zielonka(owner[~A], edges[np.ix_(~A, ~A)], priorities[~A])
        if (player and (~np.any(~z1))) or (not player and (~np.any(z1))):
            s[~A] = np.where(se != -1, np.where(~A)[0][se], s[~A])
            return (
                np.ones(len(owner), dtype=bool)
                if player
                else np.zeros(len(owner), dtype=bool),
                s,
            )
        else:
            x = np.zeros(len(owner), dtype=bool)
            y = np.zeros(len(owner), dtype=bool)
            x[(np.where(~A)[0])[~z1]] = True
            y[(np.where(~A)[0])[z1]] = True
            B, t = _find_attractor(1 - player, owner, edges, x if player else y)
            t[~A] = np.where(se != -1, np.where(~A)[0][se], t[~A])
            z2, te = _zielonka(owner[~B], edges[np.ix_(~B, ~B)], priorities[~B])
            t[~B] = np.where(te != -1, np.where(~B)[0][te], t[~B])
            if player:
                a = np.zeros(len(owner), dtype=bool)
                a[(np.where(~B)[0])[z2]] = True
                return a, t
            else:
                b = np.ones(len(owner), dtype=bool)
                b[(np.where(~B)[0])[~z2]] = False
                return b, t


class ParityGame(Game):
    def __init__(self, owner, edges, priorities):

        super().__init__(owner, edges)
        self.priorities = priorities

    @classmethod
    def generate(cls, n, p, h):
        assert (
            p >= 1 / n
        ), "Since |post(v)| needs to be >=1 for every v, p needs to be at least p>=1/n"
        p = max(0, min(((p * n) - 1) / (n - 1), 1))
        owner = np.random.choice([False, True], size=n)
        edges = np.zeros((n, n), dtype=bool)
        edges[np.arange(n), np.random.randint(0, n, n)] = True
        edges[edges == False] = np.random.choice(
            [False, True], size=n * (n - 1), p=[1 - p, p]
        )
        priorities = np.random.randint(0, h, size=n)
        return cls(owner, edges, priorities)

    def _solve_both_zielonka(self):

        prios_k = np.argsort(self.priorities)
        steps = self.priorities[prios_k][1:] - self.priorities[prios_k][:-1]
        tmp = self.priorities.copy()
        for n in np.where(steps >= 2)[0]:
            tmp[prios_k[n + 1 :]] -= (steps[n] // 2) * 2

        return _zielonka(self.owner, self.edges, tmp)

    def _to_mpg(self):

        prios_k = np.argsort(self.priorities)
        prios = self.priorities[prios_k]
        prios_even = prios % 2 == 0
        weight = np.ones(len(self.owner), dtype=int)

        for i, pr in enumerate(prios):
            if pr % 2 == 0:
                weight[i] = np.sum(weight[:i][~prios_even[:i]])
            else:
                weight[i] = np.sum(weight[:i][prios_even[:i]]) + 1

        weight[~prios_even] = -weight[~prios_even]

        y = np.empty(len(self.owner), dtype=int)
        y[prios_k] = weight

        edges = np.where(self.edges, np.tile(y.reshape(-1, 1), len(self.owner)), mini)

        return MeanPayoffGame(self.owner, edges)

    def solve_value_mpg(self):

        prios_k = np.argsort(self.priorities)
        steps = self.priorities[prios_k][1:] - self.priorities[prios_k][:-1]
        tmp = self.priorities.copy()
        for n in np.where(steps >= 2)[0]:
            tmp[prios_k[n + 1 :]] -= (steps[n] // 2) * 2

        g = ParityGame(self.owner, self.edges, tmp)

        mpg = g._to_mpg()

        v = mpg.solve_value()

        return v < 0

    def solve_strat_mpg(self):

        prios_k = np.argsort(self.priorities)
        steps = self.priorities[prios_k][1:] - self.priorities[prios_k][:-1]
        tmp = self.priorities.copy()
        for n in np.where(steps >= 2)[0]:
            tmp[prios_k[n + 1 :]] -= (steps[n] // 2) * 2

        g = ParityGame(self.owner, self.edges, tmp)

        mpg = g._to_mpg()

        return mpg.solve_strat()

    def solve_value(self, strat=None):

        if type(strat) != type(None):

            edges = (strat == -1).reshape(-1, 1) * self.edges

            for i in np.where(strat != -1)[0]:
                edges[i, strat[i]] = True

            return ParityGame(
                self.owner, edges, self.priorities
            )._solve_both_zielonka()[0]

        else:

            return self._solve_both_zielonka()[0]

    def solve_strat(self):

        return self._solve_both_zielonka()[1]

    def solve_both(self):

        return self._solve_both_zielonka()

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
        view.attr(bgcolor="#f0f0f0", forcelabels="True")
        for i, (owner, priorities) in enumerate(zip(self.owner, self.priorities)):
            if type(values) != type(None):
                label = f"<v(v<sub>{i}</sub>)={e_o[values[i]]}>"
            else:
                label = f"<v<sub>{i}</sub>>"
            view.node(
                f"{i}",
                label=label,
                xlabel=f"{priorities}",
                shape=shape[owner],
                fontcolor=colour[owner][strat[i] != -1],
                color=colour[owner][strat[i] != -1],
            )
        idx = np.where(self.edges == True)
        for s, t in zip(idx[0], idx[1]):
            view.edge(
                str(s),
                str(t),
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
                assert typee == "pg"
                owner = file.readline().replace("\n", "")
                owner = owner.split(",")
                owner = np.array(
                    [True if (e == "1" or e == "True") else False for e in owner]
                )
                priorities = file.readline().replace("\n", "")
                priorities = priorities.split(",")
                priorities = np.array([int(e) for e in priorities])
                edges = file.read().split("\n")
                edges = [e.split(",") for e in edges]
                edges = np.array(
                    [[True if f == "1" else False for f in e] for e in edges]
                )
            return ParityGame(owner, edges, priorities)

    def save_csv(self, target_path=None):
        if target_path == None:
            target_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "graphs",
                f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv",
            )
        with open(target_path, "w") as file:
            file.write("pg\n")
            file.write(",".join(["1" if e else "0" for e in self.owner]) + "\n")
            file.write(",".join([str(e) for e in self.priorities]) + "\n")
            file.write(
                "\n".join([",".join(["1" if f else "" for f in e]) for e in self.edges])
            )
        return target_path

import os.path
from datetime import datetime
import pickle
import numpy as np
from numba import jit

dark = "3f"
middle = "78"
bright = "cf"

colour = {
    0: {False: f"#00{dark}00", True: f"#00{bright}00"},
    1: {False: f"#{dark}0000", True: f"#{bright}0000"},
    2: {False: f"#0000{middle}", True: f"#0000{middle}"},
}
shape = {0: "square", 1: "circle", 2: "diamond"}

# mini, maxi, np.nan are used to indicate "edge does not exist" or similar indavlid values in incidence matrices
mini = np.iinfo(int).min
maxi = np.iinfo(int).max


def printm(edges):

    mini = np.iinfo(edges.dtype).min
    maxi = np.iinfo(edges.dtype).max

    print(
        np.where(
            edges == mini,
            "-",
            np.where(edges == maxi, "+", np.where(edges == np.nan, "x", edges)),
        )
    )


@jit(nopython=True, cache=True)
def numbafy(x, op, axis):
    out = np.empty(x.shape[1 - axis], dtype=x.dtype)
    for i in range(x.shape[1 - axis]):
        if axis:
            out[i] = op(x[i])
        else:
            out[i] = op(x[:, i])
    return out


@jit(nopython=True, cache=True)
def numba_argmax_axis1(x):
    out = np.empty(x.shape[0], dtype=np.int32)
    for i in range(x.shape[0]):
        out[i] = np.argmax(x[i])
    return out


@jit(nopython=True, cache=True)
def numba_any_axis1(x):
    out = np.zeros(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_or(out, x[:, i])
    return out


class Game:
    def __init__(self, owner, edges):

        self.owner = owner
        self.edges = edges

    @classmethod
    def generate(cls, n, p, w):
        assert (
            p >= 1 / n
        ), "Since |post(v)| needs to be >=1 for every v, p needs to be at least p>=1/n"
        # truncating o to [0,1] because otherwise rounding error can put it <0 or >1
        p = max(0, min(((p * n) - 1) / (n - 1), 1))
        owner = np.random.choice([False, True], size=n)
        edges = np.full((n, n), mini)
        edges[np.arange(n), np.random.randint(0, n, n)] = np.random.randint(
            -w, w + 1, size=n
        )
        e = np.random.choice([False, True], size=n * (n - 1), p=[1 - p, p])
        edges[edges == mini] = np.where(
            e, np.random.randint(-w, w + 1, size=n * (n - 1)), mini
        )
        return owner, edges

    def save_bin(self, target_path=None):

        if target_path == None:
            target_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "graphs",
                f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.bin",
            )

        with open(target_path, "wb") as file:
            pickle.dump(self, file)

        return target_path

    @staticmethod
    def load_bin(target_path):
        if os.path.isfile(target_path):
            with open(target_path, "rb") as file:
                try:
                    game = pickle.load(file)
                    return game
                except Exception as e:
                    print(e)

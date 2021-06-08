import numpy as np
from graphviz import Digraph
from fractions import Fraction

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

        # TODO Fig. 3

class SimpleStochasticGame:

    def __init__(self, vertices, owner, edges, avg_chance):

        self.vertices = vertices
        self.owner = owner        
        self.edges = edges
        self.avg_chance = avg_chance

if __name__ == '__main__':

    N = 10

    # pg demo
    vertices = N
    owner = np.random.choice([False, True], size=(vertices))
    edges = np.random.choice([False, True], size=(vertices,vertices))
    priority = np.random.randint(0, vertices, size=(vertices))
    pg = ParityGame(vertices,owner,edges,priority)
    print(pg)

    mpg = pg.to_mpg()
    print(mpg)

    # mpg demo
    vertices = N
    owner = np.random.choice([False, True], size=(vertices))
    edges_exist = np.random.choice([False, True], size=(vertices,vertices))
    edges_value = np.random.randint(-10, 11, size=(vertices,vertices))
    mini = np.iinfo(edges_value.dtype).min
    edges = np.where(edges_exist, edges_value, mini)
    threshold = np.random.randint(-vertices,vertices, size=(1))
    mpg = MeanPayoffGame(vertices, owner, edges, threshold)
    print(mpg)

    dpg = mpg.to_dpg()
    print(dpg)

    # dpg demo
    vertices = N
    owner = np.random.choice([False, True], size=(vertices))
    edges_exist = np.random.choice([False, True], size=(vertices,vertices))
    edges_value = np.random.randint(-10, 11, size=(vertices,vertices))
    mini = np.iinfo(edges_value.dtype).min
    edges = np.where(edges_exist, edges_value, mini)
    threshold = np.random.randint(-vertices,vertices, size=(1))
    discount = np.random.rand(1)
    dpg = DiscountedMeanPayoffGame(vertices, owner, edges, threshold, discount)
    print(dpg)

    ssg = dpg.to_ssg()
    print(ssg)

    # ssg demo
    vertices = N
    owner = np.random.randint(0, 3, size=(vertices), dtype=np.int8)
    edges = np.random.choice([False, True], size=(vertices,vertices+2))
    avg_n = len(np.where(owner==2)[0])
    avg_chance = np.random.randint(0, 256, size=(avg_n,vertices+2), dtype=np.uint8)
    avg_chance = np.where(edges[np.where(owner==2)], avg_chance, 0)
    ssg = SimpleStochasticGame(vertices, owner, edges, avg_chance)
    print(ssg)

    # d = {False:"P0",True:"P1"}
    # d2 = {False:"square",True:"circle"}

    # view = Digraph(format="png")

    # for v in range(vertices):
    #     view.node(str(v), d[owner[v]], shape=d2[owner[v]])
    # print(edges_value.shape)
    # for i in range(edges_value.shape[0]):
    #     for j in range(edges_value.shape[0]):
    #         if edges_exist[i,j]:
    #             view.edge(str(i),str(j),str(edges_value[i,j]))

    # view.render(filename=r"C:\Users\Maxime\Desktop\test.png", view=True, cleanup=True)

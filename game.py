import os.path
from datetime import datetime
import pickle

colour = {"green":{"dark":"#005f00","bright":"#00df00"},"red":{"dark":"#5f0000","bright":"#df0000"},"blue":"#00009f"}

class Game:

    def __init__(self, owner, edges):

        self.owner = owner
        self.edges = edges

    def save(self, target_path=None):

        if target_path == None:
            target_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "graphs", f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.bin")

        with open(target_path, "wb") as file:
            pickle.dump(self, file)

        return target_path

    @staticmethod
    def load(target_path):
        if os.path.isfile(target_path):
            with open(target_path, "rb") as file:
                try:
                    game = pickle.load(file)
                    return game
                except Exception as e:
                    print(e)

    def printm(self):

        mini = np.iinfo(self.edges.dtype).min
        maxi = np.iinfo(self.edges.dtype).max

        print(np.where(self.edges==mini,"-",np.where(self.edges==maxi,"+",self.edges)))
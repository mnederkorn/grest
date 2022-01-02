import os.path
from datetime import datetime
import pickle
import numpy as np

dark = "3f"
middle = "78"
bright = "cf"

colour = {0:{False:f"#00{dark}00",True:f"#00{bright}00"},1:{False:f"#{dark}0000",True:f"#{bright}0000"},2:{False:f"#0000{middle}",True:f"#0000{middle}"}}
shape = {0:"square", 1:"circle", 2:"diamond"}

def printm(edges):

    mini = np.iinfo(edges.dtype).min
    maxi = np.iinfo(edges.dtype).max
    
    print(np.where(edges==mini,"-",np.where(edges==maxi,"+",np.where(edges==np.nan,"x",edges))))

class Game:

    def __init__(self, owner, edges):

        self.owner = owner
        self.edges = edges

    def save(self, target_path=None):

        if target_path == None:
            target_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "graphs", f"{self.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.bin")

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
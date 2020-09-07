from abc import ABC, abstractmethod

class DataClass(ABC):
    @abstractmethod
    def eval(self,x):
        pass
    @abstractmethod
    def grad(self,x):
        pass
    @abstractmethod
    def draw(self,x):
        pass
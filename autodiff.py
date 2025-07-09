import numpy as np
import numpy.typing as npt
from typing import Union, Sequence

class autodiffArray:
    def __init__(self, value: Union[npt.NDArray, "autodiffArray"], grad=0.0):
        # autodiffGraph().add_node(self)
        if isinstance(value, np.ndarray):
            self.parent1 = None
            self.parent2 = None
            self.parentOperator = None
            self.value = value
        elif isinstance(value, autodiffArray):
            self.parent1 = value
            self.parent2 = None
            self.parentOperator = None
            self.value = value.value
        else:
            raise TypeError("Value must be a numpy array or an autodiffArray instance.")

        self.grad = grad

    def __add__(self, other):
        if isinstance(other, autodiffArray):
            autodiffArray(self.value + other.value, self.grad + other.grad)

        else:
            return autodiffArray(self.value + other, self.grad)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, autodiffArray):
            out = autodiffArray(self.value - other.value, self.grad - other.grad)
            out.parent1 = self
            out.parent2 = other
            out.parentOperator = '-'
            return out
        else:
            out = autodiffArray(self.value - other, self.grad - other.grad)
            out.parent1 = self
            out.parent2 = None # I think you don't need to save the second parent if it's not an autodiffArray for subtraction
            out.parentOperator = '-'
            return out

    def __rsub__(self, other):
        return autodiffArray(other - self.value, -self.grad)

    def __mul__(self, other):
        if isinstance(other, autodiffArray):
            out = autodiffArray(self.value * other.value, self.grad * other.value + self.value * other.grad)
            out.parent1 = self
            out.parent2 = other
            out.parentOperator = '*'
            return out
        else:
            out = autodiffArray(self.value * other, self.grad * other.value + self.value * other.grad)
            out.parent1 = self
            out.parent2 = other
            out.parentOperator = '*'
            return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, autodiffArray):
            return autodiffArray(self.value / other.value,
                            (self.grad * other.value - self.value * other.grad) / (other.value ** 2))
        else:
            return autodiffArray(self.value / other, self.grad / other)

    def __rtruediv__(self, other):
        return autodiffArray(other / self.value, -other * self.grad / (self.value ** 2))

    def __repr__(self):
        return f"autodiffArray(value={self.value}, grad={self.grad})"


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

# class autodiffGraph(metaclass=Singleton):
#     def __init__(self):
#         self.nodes = []

#     def add_node(self, node, parent1=None, parent2=None):
#         if not isinstance(node, autodiffArray):
#             raise TypeError("Only autodiffArray instances can be added to the graph.")
#         if node in self.nodes:
#             return
#         self.nodes.append(node)
#         if parent1 is not None:
#             self.add_node(parent1)
#         if parent2 is not None:
#             self.add_node(parent2)

        


#     def compute_gradients(self, from_node):
#         for node in reversed(self.nodes):
#             if isinstance(node, autodiffArray):
#                 # Here we would compute the gradients based on the operations
#                 # For simplicity, we assume that the gradients are already set
#                 pass



# def test1():
#     graph = autodiffGraph()
#     a = autodiffArray(np.array(2.0), 1.0)
#     b = autodiffArray(np.array(3.0), 1.0)
#     c = a + b
#     d = c * 2
#     e = d / 4
#     graph.compute_gradients(e)
#     print(e)



if __name__ == "__main__":
    # graph = autodiffGraph()
    d = autodiffArray(np.array(5.0), 1.0)
    e = autodiffArray(d * 2, 1.0)


    print(hex(id(e)))
    e = autodiffArray(e * 2, 1.0)
    print(hex(id(e)))
import numpy as np
import numpy.typing as npt
from typing import Union, Sequence

class autodiffArray:
    def __init__(self, value):
        
        if isinstance(value, autodiffArray):
            self.parent1 = value
            self.parent2 = None
            self.parentOperator = None
            self.value = value.value
        else:
            
            self.parent1 = None
            self.parent2 = None
            self.parentOperator = None
            self.value = value
        self.grad = 0

    def __add__(self, other):
        if not isinstance(other, autodiffArray):
            other = autodiffArray(other)
        out = autodiffArray(self.value + other.value)
        out.parent1 = self
        out.parent2 = other # I think you don't need to save the second parent if it's not an autodiffArray for subtraction
        out.parentOperator = '+'
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, autodiffArray):
            other = autodiffArray(other)
        out = autodiffArray(self.value - other.value)
        out.parent1 = self
        out.parent2 = other # I think you don't need to save the second parent if it's not an autodiffArray for subtraction
        out.parentOperator = '-'
        return out

    def __rsub__(self, other):
        return autodiffArray(other - self.value)

    def __mul__(self, other):
        if not isinstance(other, autodiffArray):
            other = autodiffArray(other)

        out = autodiffArray(self.value * other.value)
        out.parent1 = self
        out.parent2 = other
        out.parentOperator = '*'
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, autodiffArray):
            other = autodiffArray(other)
        out = autodiffArray(self.value / other.value)
        out.parent1 = self
        out.parent2 = other
        out.parentOperator = '/'
        
        return out

    def __rtruediv__(self, other):
        return autodiffArray(other / self.value)

    def __repr__(self):
        return f"autodiffArray(value={self.value}, grad={self.grad})"

    def backward(self):
        self.grad = 1.0
        self.backward_helper()
    def backward_helper(self):
        if not isinstance(self.parent1, autodiffArray) or not isinstance(self.parent2, autodiffArray):
            return
        
        if self.parentOperator == '*':
            self.parent1.grad += self.grad * self.parent2.value
            self.parent2.grad += self.grad * self.parent1.value
        elif self.parentOperator == '+':
            self.parent1.grad += self.grad
            self.parent2.grad += self.grad
        elif self.parentOperator == '-':
            self.parent1.grad += self.grad
            self.parent2.grad -= self.grad
        elif self.parentOperator == '/':
            self.parent1.grad += self.grad / self.parent2.value
            self.parent2.grad -= self.grad / self.parent2.value ** 2 * self.parent1.value
        
        self.parent1.backward_helper()
        if self.parent1 != self.parent2: # double counts if they are the same (grad was already twice as much as it should be)
            self.parent2.backward_helper()
            

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



def test1():
    a = autodiffArray(np.array([2.0]))
    b = autodiffArray(np.array([3.0]))
    c = a * b
    d = c * 2
    e = d / 4
    # graph.compute_gradients(e)
    # print(e)
    e.backward()
    print(a.grad, b.grad)

def test_addition_backward_fixed():
    a = autodiffArray(np.array([3.0]))
    b = autodiffArray(np.array([7.0]))
    c = a + b
    c.backward()
    assert np.allclose(a.grad, np.array([1.0]))
    assert np.allclose(b.grad, np.array([1.0]))

def test_subtraction_backward_fixed():
    a = autodiffArray(np.array([10.0]))
    b = autodiffArray(np.array([4.0]))
    c = a - b
    c.backward()
    assert np.allclose(a.grad, np.array([1.0]))
    assert np.allclose(b.grad, np.array([-1.0]))

def test_multiplication_backward_fixed():
    a = autodiffArray(np.array([5.0]))
    b = autodiffArray(np.array([2.0]))
    c = a * b
    c.backward()
    assert np.allclose(a.grad, np.array([2.0]))
    assert np.allclose(b.grad, np.array([5.0]))

def test_division_backward_fixed():
    a = autodiffArray(np.array([8.0]))
    b = autodiffArray(np.array([2.0]))
    c = a / b
    c.backward()
    assert np.allclose(a.grad, np.array([1.0 / 2.0]))
    assert np.allclose(b.grad, np.array([-8.0 / (2.0 ** 2)]))

def test_addition_backward_random():
    a_val = np.random.randn(1)
    b_val = np.random.randn(1)
    a = autodiffArray(a_val)
    b = autodiffArray(b_val)
    c = a + b
    c.backward()
    assert np.allclose(a.grad, np.array([1.0]))
    assert np.allclose(b.grad, np.array([1.0]))

def test_subtraction_backward_random():
    a_val = np.random.randn(1)
    b_val = np.random.randn(1)
    a = autodiffArray(a_val)
    b = autodiffArray(b_val)
    c = a - b
    c.backward()
    assert np.allclose(a.grad, np.array([1.0]))
    assert np.allclose(b.grad, np.array([-1.0]))

def test_multiplication_backward_random():
    a_val = np.random.randn(1)
    b_val = np.random.randn(1)
    a = autodiffArray(a_val)
    b = autodiffArray(b_val)
    c = a * b
    c.backward()
    assert np.allclose(a.grad, b_val)
    assert np.allclose(b.grad, a_val)

def test_division_backward_random():
    a_val = np.random.randn(1)
    b_val = np.random.randn(1)
    # Avoid division by zero
    b_val[b_val == 0] = 1.0
    a = autodiffArray(a_val)
    b = autodiffArray(b_val)
    c = a / b
    c.backward()
    assert np.allclose(a.grad, 1.0 / b_val)
    assert np.allclose(b.grad, -a_val / (b_val ** 2))

def test_nested_operations_backward():
    a = autodiffArray(np.array([2.0]))
    b = autodiffArray(np.array([3.0]))
    c = autodiffArray(np.array([4.0]))
    e = autodiffArray(np.array([5.0]))
    d = ((a + b) * c) / e
    d.backward()
    # d = ((a + b) * c) / e
    # dd/da = c / e
    # dd/db = c / e
    # dd/dc = (a + b) / e
    # dd/de = -((a + b) * c) / (e ** 2)
    assert np.allclose(a.grad, np.array([4.0 / 5.0]))
    assert np.allclose(b.grad, np.array([4.0 / 5.0]))
    assert np.allclose(c.grad, np.array([5.0 / 5.0]))
    assert np.allclose(e.grad, np.array([-20.0 / 25.0]))

def test_nested_operations_reassign():
    a = autodiffArray(np.array([-2.0]))
    b = autodiffArray(np.array([3.0]))

    c = a + 2 * b
    e = c * c
    e.backward()
    print("test1",a.grad, b.grad)
    assert np.allclose(a.grad, np.array([8]))
    assert np.allclose(b.grad, np.array([16]))


    
    a = autodiffArray(np.array([-2.0]))
    b = autodiffArray(np.array([3.0]))

    c = (a + 2) * b
    c = c * c
    c.backward()
    print("test2",a.grad, b.grad)
    assert np.allclose(a.grad, np.array([0]))
    assert np.allclose(b.grad, np.array([0]))

if __name__ == "__main__":
    test1()
    test_addition_backward_fixed()
    test_subtraction_backward_fixed()
    test_multiplication_backward_fixed()
    test_division_backward_fixed()
    test_addition_backward_random()
    test_subtraction_backward_random()
    test_multiplication_backward_random()
    test_division_backward_random()
    test_nested_operations_backward()
    test_nested_operations_reassign()
    # d = autodiffArray(np.array([5.0]), 1.0)
    # d2 = d * 2
    # print(d2)
    # e = autodiffArray(d2, 1.0)


    # print(hex(id(e)))
    # e = autodiffArray(e * 2, 1.0)
    # print(hex(id(e)))
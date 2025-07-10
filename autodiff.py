import numpy as np
import numpy.typing as npt
from typing import Union, Sequence
from tensorflow.keras.datasets import mnist

class autodiffArray:
    def __init__(self, value):
        
        if isinstance(value, autodiffArray):
            self.parent1 = value
            self.parent2 = None
            self.parentOperator = None
            self.value = value.value
        # elif isinstance(value, (int, float)):
            
        #     self.parent1 = None
        #     self.parent2 = None
        #     self.parentOperator = None
        #     self.value = np.array([value])
        else:
            self.parent1 = None
            self.parent2 = None
            self.parentOperator = None
            self.value = value
        self.grad = 0.0# np.zeros_like(self.value,  dtype='float64')  # Initialize gradient to zero

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
        if not isinstance(other, autodiffArray):
            other = autodiffArray(other)
        return other.__truediv__(self)

    def __matmul__(self, other):
        if not isinstance(other, autodiffArray):
            other = autodiffArray(other)
        out = autodiffArray(self.value @ other.value)
        out.parent1 = self
        out.parent2 = other
        out.parentOperator = '@'
        return out
    
    def __pow__(self, exponent):
        out = self
        for _ in range(exponent - 1):
            out = out * self
        return out

    def __repr__(self):
        return f"autodiffArray(value={self.value}, grad={self.grad})"

    def backward(self):
        self.grad = np.ones_like(self.value, dtype=np.float64)  # Initialize gradient to 1 for the output node
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
        elif self.parentOperator == '@':
            print("backward matmul")
            
            i = [0]
            def tryit(func):
                i[0] += 1
                try:
                    print(i[0], func())
                except:
                    pass
            tryit(lambda: self.grad.shape)
            tryit(lambda: self.parent1.value.shape)
            tryit(lambda: self.parent2.value.shape)
            tryit(lambda: self.parent1.grad.shape)
            tryit(lambda: self.parent2.grad.shape)
            tryit(lambda: self.grad)
            tryit(lambda: self.parent1.value)
            tryit(lambda: self.parent2.value)
            tryit(lambda: self.parent1.grad)
            tryit(lambda: self.parent2.grad)
            
            self.parent1.grad += self.grad @ self.parent2.value.T
            self.parent2.grad += (self.grad.T @ self.parent1.value).T

        self.parent1.backward_helper()
        if self.parent1 != self.parent2: # double counts if they are the same (grad was already twice as much as it should be)
            self.parent2.backward_helper()
            
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        self.biases = []
        for i in range(len(layer_sizes)-1):
            self.layers.append(autodiffArray(np.random.randn(layer_sizes[i], layer_sizes[i+1])))
            self.biases.append(autodiffArray(np.random.randn(layer_sizes[i+1])))
    def forward(self, x):
        for layer, bias in zip(self.layers, self.biases):
            x = x @ layer + bias.value
        return x

    def zero_grad(self):
        for layer in self.layers:
            layer.grad = 0
        for bias in self.biases:
            bias.grad = 0

    def update(self, learning_rate=0.01):
        for i in range(len(self.layers)):
            self.layers[i].value -= learning_rate * self.layers[i].grad
            self.biases[i].value -= learning_rate * self.biases[i].grad
            self.zero_grad()
        


# def autodiffSum(arr: autodiffArray):
#     arr.value = np.sum(arr.value)
#     arr.grad = np.on


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
    a = autodiffArray(np.array([-2.0, 3]))
    b = autodiffArray(np.array([3.0, -1]))

    c = a + 2 * b
    e = c * c
    e.backward()
    print("test1",a.grad, b.grad)
    assert np.allclose(a.grad, np.array([8, 2]))
    assert np.allclose(b.grad, np.array([16,4]))


    
    a = autodiffArray(np.array([-2.0]))
    b = autodiffArray(np.array([3.0]))

    c = (a + 2) * b
    c = c * c
    c.backward()
    print("test2",a.grad, b.grad)
    assert np.allclose(a.grad, np.array([0]))
    assert np.allclose(b.grad, np.array([0]))


def test_matmul():
    a = autodiffArray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = autodiffArray(np.array([[5.0, 6.0], [7.0, 8.0]]))
    c = a @ b
    c.backward()
    assert np.allclose(a.grad, np.array([[5.0 + 14.0, 6.0 + 16.0], [7.0 + 24.0, 8.0 + 32.0]]))
    assert np.allclose(b.grad, np.array([[1.0 * 3.0 + 2.0 * 4.0, 1.0 * 4.0 + 2.0 * 8.0], [3.0 * 3.0 + 4.0 * 4.0, 3.0 * 4.0 + 4.0 * 8.0]]))

def load_mnist(normalize=True, flatten=True):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
    if flatten:
        x_train = x_train.reshape(-1, 28*28)
        x_test = x_test.reshape(-1, 28*28)
    return (x_train, y_train), (x_test, y_test)

def testTrainMnist():
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train = autodiffArray(x_train)
    y_train = autodiffArray(np.eye(10)[y_train])
    nn = NeuralNetwork([784, 128, 10])  # Example neural network with input size 784 (28x28), hidden layer of size 128, and output layer of size 10
    print("MNIST data loaded and converted to autodiffArray.")
    for i in range(10):
        x = x_train
        y = y_train
        y_hat = nn.forward(x)
        # Here you can add more operations and test gradients
        # For example, you can compute a loss and call backward
        sumAlongRows = autodiffArray(np.ones([1, y.value.shape[0]])) @ ((y_hat - y) ** 2)
        # sumAlongRows.value = sumAlongRows.value[None, :]
        print("sum shapes",sumAlongRows.value.shape)
        loss = (sumAlongRows @ autodiffArray(np.ones([y.value.shape[1], 1]))) / (60000 * 100)
        # loss2 =  (((y_hat - y) ** 2) @ autodiffArray(np.ones([y.value.shape[0]]))) @ autodiffArray(np.ones([y.value.shape[1]]        print(loss2.value, loss.value)
        print(loss.value.shape)
        loss.backward()
        nn.update()
        # print(f"Gradients: x.grad={x.grad}, y.grad={y.grad}")
        print("epoch", i,"loss:",loss.value)
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
    testTrainMnist()
    # d = autodiffArray(np.array([5.0]), 1.0)
    # d2 = d * 2
    # print(d2)
    # e = autodiffArray(d2, 1.0)


    # print(hex(id(e)))
    # e = autodiffArray(e * 2, 1.0)
    # print(hex(id(e)))
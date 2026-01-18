import numpy as np
import numpy.typing as npt
from typing import Union, Sequence
from tensorflow.keras.datasets import mnist
import sys
import matplotlib.pyplot as plt

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
        self.grad = 0.0 # Initialize gradient to zero

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
            self.parent1.grad += self.grad @ self.parent2.value.T
            self.parent2.grad += (self.grad.T @ self.parent1.value).T
        elif self.parentOperator == 'relu':
            # ReLU derivative is 1 for positive values, 0 for negative values
            relu_grad = np.where(self.parent1.value > 0, 1, 0)
            self.parent1.grad += self.grad * relu_grad # parent one and two will always be the same for relu
        self.parent1.backward_helper()
        if self.parent1 != self.parent2: # double counts if they are the same (grad was already twice as much as it should be)
            self.parent2.backward_helper()
    def zero_grad(self):
        self.grad = 0.0
        if isinstance(self.parent1, autodiffArray):
            self.parent1.zero_grad()
        if isinstance(self.parent2, autodiffArray):
            self.parent2.zero_grad()


def relu(activation):
    """
    Applies the ReLU activation function element-wise.
    """
    out = autodiffArray(np.maximum(0, activation.value))
    out.parent1 = activation
    out.parent2 = activation
    out.parentOperator = 'relu'
    return out
            
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        self.biases = []
        for i in range(len(layer_sizes)-1):
            self.layers.append(autodiffArray(np.random.randn(layer_sizes[i], layer_sizes[i+1]) / np.sqrt(layer_sizes[i] / 2)))
            self.biases.append(autodiffArray(np.random.randn(layer_sizes[i+1])))
    def forward(self, x):
        for i, (layer, bias) in enumerate(zip(self.layers, self.biases)):
            x = x @ layer + bias
            if i < len(self.layers) - 1:
                x = relu(x)
        return x

    def zero_grad(self):
        for layer in self.layers:
            layer.grad = 0
        for bias in self.biases:
            bias.grad = 0
        

    def update(self, learning_rate=0.01):
        for i in range(len(self.layers)):
            self.layers[i].value -= learning_rate * self.layers[i].grad
            self.biases[i].value -= learning_rate * np.average(self.biases[i].grad, axis=0) if isinstance(self.biases[i].grad, np.ndarray) else learning_rate * self.biases[i].grad
        self.zero_grad()

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

def getAcc(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

def getMinibatch(x, y, batch_size=100):
    indices = np.random.choice(x.shape[0], batch_size, replace=False)
    x_batch = x[indices]
    y_batch = y[indices]
    return autodiffArray(x_batch), autodiffArray(y_batch)

def avg(arr: autodiffArray):
    sumAlongRows = autodiffArray(np.ones([1, arr.value.shape[0]])) @ arr
    return (sumAlongRows @ autodiffArray(np.ones([arr.value.shape[1], 1]))) / (arr.value.shape[0] * arr.value.shape[1])

def testTrainMnist():
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_test = autodiffArray(x_test)
    y_test = autodiffArray(np.eye(10)[y_test]) # Convert labels to one-hot
    x_train = x_train
    y_train = np.eye(10)[y_train]
    nn = NeuralNetwork([784, 128, 10])  # Example neural network with input size 784 (28x28), hidden layer of size 128, and output layer of size 10
    print("MNIST data loaded and converted to autodiffArray.")
    test_losses = []
    for i in range(100000):
        x, y = getMinibatch(x_train, y_train, batch_size=128)
        y_hat = nn.forward(x)
        # Here you can add more operations and test gradients
        # For example, you can compute a loss and call backward
        
        loss = avg((y_hat - y) ** 2)
       
        
        loss.backward()
        
        if i % 1000 == 0:
            print("train acc:", getAcc(y.value, y_hat.value))
            print("epoch", i,"loss:",loss.value)
       
        nn.update(learning_rate=0.01)
        if i % 10 == 0:
            y_test_hat = nn.forward(x_test)
            test_loss = avg((y_test_hat - y_test) **2) 
            # print(f"Epoch {i}, Test Loss: {test_loss.value[0]}")
            test_losses.append(test_loss.value[0])
        nn.zero_grad() # paranoia
    
    plt.plot(test_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Test Loss')
    plt.title('Test Loss Over Time')
    plt.show()
        
    
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
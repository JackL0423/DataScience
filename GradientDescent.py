from LinearAlgebra import Vector, dot

def sum_of_squares(v: Vector) -> float:
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

from typing import Callable

def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return (f(x + h) - f(x)) / h 

def partial_difference_quotient(f: Callable[[Vector], float],
                                v: Vector,
                                i: int,
                                h: float) -> float:
    """Returns the i-th partial difference quotient of f at v"""
    w = [v_j + (h if j == i else 0)     # add h to just the ith element of v
         for j, v_j in enumerate(v)]
    
    return(f(w) - f(v)) / h

def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float=0.0001):
    return [partial_difference_quotient(f, v, i, h)
            for i in range(len(v))]

import random
from LinearAlgebra import distance, add, scalar_multiply

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Moves 'step_size' in the 'gradient' direction from 'v' """
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v: Vector)-> Vector:
    return [2 * v_i for v_i in v]

#v = [random.uniform(-10, 10) for i in range(3)]

#for epoch in range(1000):
#    grad = sum_of_squares_gradient(v)
#    v = gradient_step(v, grad, -0.01)
#    print(epoch, v)

#assert distance(v, [0, 0, 0]) < 0.001

inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept
    error = (predicted - y)
    squared_error = error ** 2
    grad = [2 * error * x, 2 * error]
    return grad

from LinearAlgebra import vector_mean

# start with random values for slope and intercept
#theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

learning_rate = 0.001

#for epoch in range(5000):
    # compute the mean of the gradients
#    grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
    # take a step in that direction
#    theta = gradient_step(theta, grad, -learning_rate)
#    print(epoch, theta)

#slope, intercept = theta
#assert 19.9 < slope < 20.1,   "slope should be about 20"
#assert 4.9 < intercept < 5.1, "intercept should be about 5"

from typing import TypeVar, List, Iterator

T = TypeVar('T')    # allows us to type 'generic' functions

def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool=True) -> Iterator[List[T]]:
    """Generates 'batch_size' -sized minibatches from the dataset"""
    # start to indexes 0, batch_size, 2 * batch_size, ...
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle: random.shuffle(batch_starts)    # shuffle the batches

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]

theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

#for epoch in range(1000):
#    for batch in minibatches(inputs, batch_size=20):
#        grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
#        theta = gradient_step(theta, grad, -learning_rate)
    #print(epoch, grad)

#slope, intercept = theta
#assert 19.9 < slope < 20.1, "slope should be about 20"
#assert 4.9 < intercept < 5.1, "intercept should be about 5"

# Another variation called stochastic gradient descent (more efficient)
#for epoch in range(100):
#    for x, y in inputs:
#        grad = linear_gradient(x, y, theta)
#        theta = gradient_step(theta, grad, -learning_rate)
#    print(epoch, theta)

#slope, intercept = theta
#assert 19.9 < slope < 20.1, "slope should be about 20"
#assert 4.9 < intercept < 5.1, "intercept should be about 5"

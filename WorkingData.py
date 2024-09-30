from typing import List, Dict
from collections import Counter
import math

import matplotlib.pyplot as plt 

def bucketsize(point: float, bucket_size: float) -> float:
    """Floor the point to the next lower multiple of bucket_size"""
    return bucket_size * math.floor(point / bucket_size)

def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
    """Buckets the points and counts how many in each bucket"""
    return Counter(bucketsize(point, bucket_size) for point in points)

def plot_histogram(points: List[float], bucket_size: float, title: str=""): 
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)

import random
from Probability import inverse_normal_cdf

random.seed(0)

# uniform between -100 and 100
# uniform = [200 * random.random() - 100 for _ in range(10000)]

# normal distribution with mean 0, standard deviation 57
# normal = [57 * inverse_normal_cdf(random.random()) for _ in range(10000)]

def random_normal() -> float:
    """Returns a random draw from a standard deviation distribution"""
    return inverse_normal_cdf(random.random())
from Statistics import correlations
from LinearAlgebra import Matrix, Vector, make_matrix

def correlation_matrix(data: List[Vector]) -> Matrix:
    """
    Returns the len(data) x len(data) matrix whose (i, j)-th entry is
    the correlation between data[i] and data[j]
    """
    def correlation_ij(i: int, j: int) -> float:
        return correlations(data[i], data[j])
    
    return make_matrix(len(data), len(data), correlation_ij)


# Using NamedTuples
import datetime

stock_price = {'closing_price': 102.06, 'date': datetime.date(2014, 8, 29), 'symbol': 'AAPL'}


from collections import namedtuple 

StockPrice = namedtuple('StockPrice', ['symbol', 'date', 'closing_price'])
price = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price.symbol == 'MSFT'
assert price.closing_price == 106.03

# Dataclasses

from dataclasses import dataclass

@dataclass
class StockPrice2:
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        """It's a class, so we can add methods too"""
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']

price2 = StockPrice2('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price2.symbol == 'MSFT'
assert price2.closing_price == 106.03
assert price2.is_high_tech()

# cleaning and munging
from dateutil.parser import parse

def parse_row(row: List[str]) -> StockPrice:
    symbol, date, closing_price = row
    return StockPrice(symbol=symbol, date=parse(date).date(), closing_price=float(closing_price))

# Test function
stock = parse_row(["MSFT", "2018-12-14", "106.03"])

assert stock.symbol == "MSFT"
assert stock.date == datetime.date(2018, 12, 14)
assert stock.closing_price == 106.03

from typing import Optional
import re 

def try_parse_row(row: List[str]) -> Optional[StockPrice]:
    symbol, date_, closing_price_ = row

    # Stock symbol should be all capital letters
    if not re.match(r"^[A-Z]+$", symbol):
        return None
    
    try:
        date = parse(date_).date()
    except ValueError:
        return None
    
    try: 
        closing_price = float(closing_price_)
    except ValueError:
        return None
    
    return StockPrice(symbol, date, closing_price)


# Should return None for errors
assert try_parse_row(["MSFT0", "2018-12-14", "106.03"]) is None
assert try_parse_row(["MSFT", "2018-12--14", "106.03"]) is None
assert try_parse_row(["MSFT", "2018-12-14", "x"]) is None

# Should return same as before if data is good
assert try_parse_row(["MSFT", "2018-12-14", "106.03"]) == stock 


import csv

data: List[StockPrice] = []

#with open("comma_delimeted_stock_prices.csv") as f:
#    reader = csv.reader(f)
#    for row in reader:
#        maybe_stock = try_parse_row(row)
#        if maybe_stock is None:
#            print(f"skipping invalid row: {row}")
#        else:
#            data.append(maybe_stock)

data = [StockPrice(symbol='MSFT', date=datetime.date(2018, 12, 24), closing_price=106.03)]

#max_aapl_price = max(stock_price.closing_price for stock_price in data if stock_price.symbol == "AAPL")


from collections import defaultdict

max_prices: Dict[str, float] = defaultdict(lambda: float('-inf'))

for sp in data: 
    symbol, closing_price = sp.symbol, sp.closing_price
    if closing_price > max_prices[symbol]:
        max_prices[symbol] = closing_price

prices: Dict[str, List[StockPrice]] = defaultdict(list)

for sp in data:
    prices[sp.symbol].append(sp)

prices = {symbol: sorted(symbol_prices) for symbol, symbol_prices in prices.items()}

def pct_change(yesterday: StockPrice, today: StockPrice) -> float:
    return today.closing_price / yesterday.closing_price - 1

#from collections import NamedTuple

#class DailyChange(NamedTuple):
#    symbol: str
#    date: datetime.date
#    pct_change: float

#def day_over_changes(prices: List[StockPrice]) -> List[DailyChange]:
#    """Assume prices are for one stock and are in order"""
#    return [DailyChange(symbol=today.symbol, date=today.date, pct_change=pct_change(yesterday, today)) for yesterday, today in zip(prices, prices[1:])]

#all_changes = [change for symbol_prices in prices.values() for change in day_over_changes(symbol_prices)]

#max_change = max(all_changes, key=lambda change: change.pct_change)

#assert max_change.symbol == 'AAPL'
#assert max_change.date == datetime.date(1997, 8, 6)
#assert  0.33 < max_change.pct_change < 0.34

#changes_by_month: List[DailyChange] = {month: [] for month in range(1, 13)}

#for change in all_changes:
#    changes_by_month[change.date.month].append(change)

#avg_daily_change = {month: sum(change.pct_change for change in changes / len(changes)) for month, changes in changes_by_month.items()}

from typing import Tuple
from LinearAlgebra import distance, vector_mean
from Statistics import standard_deviation

a_to_b = distance([63, 150], [67, 160])     # 10.77
a_to_c = distance([63, 150], [70, 171])     # 22.14
b_to_c = distance([67, 160], [70, 171])     # 11.40

def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
    """Returns the mean and standard deviation for each position"""
    dim = len(data[0])

    means = vector_mean(data)
    stdevs = [standard_deviation([vector[i] for vector in data]) for i in range(dim)]

    return means, stdevs

vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]
means, stdevs = scale(vectors)
assert means == [-1, 0, 1]
assert stdevs == [2, 1, 0]

def rescale(data: List[Vector]) -> List[Vector]:
    """Rescales the input data so each position has mean 0 and standard deviation 1. Leaves position as is if its stdv is 0"""

    dim = len(data[0])
    means, stdevs = scale(data)

    # make a copy of each vector
    rescaled = [v[:] for v in data]
    
    for v in rescaled: 
        for i in range(dim):
            if stdevs[i] > 0:
                v[i] = (v[i] - means[i]) / stdevs[i]

    return rescaled

means, stdevs = scale(rescale(vectors))
assert means == [0, 0, 1]
assert stdevs == [1, 1, 0]

# tqdm - to track progress on computations that take long time
import tqdm 

for i in tqdm.tqdm(range(100)):
    # do something slow
    _ = (random.random() for _ in range(1000000))

# set description of progress bar
#def primes_up_to(n: int) -> List[int]:
#    primes = [2]

#    with tqdm.trange(3, n) as t:
#        for i in t:
            # i is prime if no smaller prime divides it
#            i_is_prime = not any(i % p == 0 for p in primes)
#            if i_is_prime:
#                primes.append(i)
            
#            t.set_description(f"{len(primes)} primes")
#    return primes

#my_primes = primes_up_to(100_000)

# Dimensional Reduction
from LinearAlgebra import subtract

def de_mean(data: List[Vector]) -> List[Vector]:
    """Recenters the data to have mean 0 in every dimension"""
    mean = vector_mean(data)
    return [subtract(vector, mean) for vector in data]


from LinearAlgebra import magnitude

def direction(w: Vector) -> Vector:
    mag = magnitude(w)
    return [w_i / mag for w_i in w]


from LinearAlgebra import dot

def directional_variance(data: List[Vector], w: Vector) -> float:
    """Returns the variance of x in the direction of w"""
    w_dir = direction(w)
    return sum(dot(v, w_dir) ** 2 for v in data)


def directional_variance_gradient(data: List[Vector], w: Vector) -> Vector:
    """The gradient of directional variance with respect to w"""
    w_dir = direction(w)
    return [sum(2 * dot(v, w_dir) * v[i] for v in data) for i in range(len(w))]


from GradientDescent import gradient_step

def first_principal_component(data: List[Vector], n: int=100, step_size: float=0.1) -> Vector:
    # start with random guess
    guess = [1.0 for _ in data[0]]

    with tqdm.trange(n) as t:
        for _ in t:
            dv = directional_variance
            gradient = directional_variance_gradient(data, guess)
            guess = gradient_step(guess, gradient, step_size)
            t.set_description(f"dv: {dv:.3f}")
            
    return direction(guess)


from LinearAlgebra import scalar_multiply

def project(v: Vector, w: Vector) -> Vector:
    """Returns the projection of v onto the direction w"""

    projection_length = dot(v, w)
    return scalar_multiply(projection_length, w)

def remove_projection_from_vector(v: Vector, w: Vector) -> Vector:
    """projects v onto w and subtracts the result from v"""
    return subtract(v, project(v, w))

def remove_projection(data: List[Vector], w: Vector) -> List[Vector]:
    return [remove_projection_from_vector(v, w) for v in data]


def pca(data: List[Vector], num_components: int) -> List[Vector]:
    components: List[Vector] = []
    for _ in range(num_components):
        component = first_principal_component(data)
        components.append(component)
        data = remove_projection(data, component)

    return components

def transform_vector(v: Vector, components: List[Vector]) -> Vector:
    return [dot(v, w) for w in components]

def transform(data: List[Vector], components: List[Vector]) -> List[Vector]:
    return [transform_vector(v, components) for v in data]


import enum, random

# Enum is typed set of enumerated numbers for better readability 
#class Kid(enum.Enum):
#    Boy = 0
#    Girl = 1

#def random_kid() -> Kid:
#    return random.choice([Kid.Boy, Kid.Girl])

both_girls = 0
older_girl = 0
either_girl = 0

#random.seed(0)

#for _ in range(10000):
#    younger = random_kid()
#    older = random_kid()
#    if older == Kid.Girl:
#        older_girl += 1
#    if older == Kid.Girl and younger == Kid.Girl:
#        both_girls += 1
#    if older == Kid.Girl or younger == Kid.Girl:
#       either_girl += 1
    
#print("P(both | older):", both_girls / older_girl)      # 0.514 ~ 1/2
#print("P(both | either):", both_girls / either_girl)    # 0.342 ~ 1/3

def uniform_cdf(x: float) -> float:
    """Returns the probability that a uniform random variable is <= x"""
    if x < 0:   return 0
    elif x < 1: return x
    else:       return 1

import math
SQRT_TWO_PI = math.sqrt(2 * math.pi)

def normal_pdf(x: float, mu: float=0, sigma: float=1) -> float:
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma))

def normal_cdf(x: float, mu: float=0, sigma: float=1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def inverse_normal_cdf(p: float,
                       mu: float=0,
                       sigma: float=1,
                       tolerance: float=0.00001) -> float:
    """Find approximate inverse using binary search"""

    # if not standard, compute standard and rescale
    if mu != 1 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    
    low_z = -10.0
    hi_z = 10.0
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z = mid_z
        else:
            hi_z = mid_z
        
    return mid_z

def bernoulli_trial(p: float) -> int:
    """Returns 1 with probability p and 0 with probability 1-p"""
    return 1 if random.random() < p else 0

def binomial(n: int, p: float) -> int:
    """Returns the sum of n bernoulli(p) trials"""
    return sum(bernoulli_trial(p) for _ in range(n))


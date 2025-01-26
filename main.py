import numpy as np
from datetime import datetime
from time import time

# Global mask initialization
mask = None

# calculate p'
def derivative(p):
    return p[:-1] * np.arange(len(p) - 1, 0, -1)

# calculate f(x)
def polyVal(f, x ):
    d = np.ones(len(f)) * x
    d[0] = 1
    d = np.multiply.accumulate(d)
    d = d[::-1]
    return np.dot(d, f)

# calculate p(x) / q(x) both polynomials
def divide(p, q, x):
    if np.abs(x) <= 1:
        return (polyVal(p, x) / polyVal(q, x))
    else:
        # Avoiding overflow case
        return x * (polyVal(p[::-1], 1 / x) / polyVal(q[::-1], 1 / x))


# F = R*cos(theta) + i * R * sin(theha) Euler equation
def eulerEquation(R, theta):
    real_part = R * np.cos(theta)
    imag_part = R * np.sin(theta)
    return real_part + 1j * imag_part

# Initialize roots
def initRoots(p):
    # number of roots is the number of coefficient of the derivative
    num_of_roots = len(p) -1
    # function to create Radius of the imaginary
    R = (1 + max(np.abs(p[1:])) / (np.abs(p[0]))) / 2
    roots = np.zeros(num_of_roots, dtype=np.complex128)
    for i in range(num_of_roots):
        theta = (2 * np.pi * i) / (num_of_roots)
        root = eulerEquation(R, theta)
        roots[i] = root
    return roots

# Initialize mask once to use at each iteration
def initGlobalMask(n):
   global mask
   mask = np.ones(n, dtype=bool)


def calculateSigma(roots, k):
   global mask
   mask[k] = False
   z_k = roots[k]
   z_j = roots[mask]
   mask[k] = True
   return np.sum(1 / (z_k - z_j))

# Calculate W equation from Aberth–Ehrlich and keeping the maximum
def calcOffset(p, p_tag, roots):
    Wmax = float('-inf')
    W = np.zeros(len(roots), dtype=np.complex128)
    for k, root in enumerate(roots):
        numerator = divide(p, p_tag, root)
        sigma = calculateSigma(roots, k)
        denominator = 1 - np.multiply(numerator, sigma)
        W[k] = np.divide(numerator, denominator)
        Wmax = max(Wmax, abs(W[k]))  # Use abs() for complex numbers
    return W, Wmax


# Aberth–Ehrlich  function to find roots.
def aberthEhrlich(p,p_tag, epsilon, max_tries):
    tries = 0
    roots = initRoots(p)
    while tries < max_tries:
        w, Wmax = calcOffset(p, p_tag, roots)
        # the alg stops when each offset is smaller than the defined epsilon
        if Wmax < epsilon:
            tries = max_tries
        # Updating roots
        roots -= w
        tries += 1
    return roots


# extract coefficients  from file
def extractCoefficients(file_name):
    with open(file_name, 'r') as file:
        return np.array([float(line.strip()) for line in file], dtype=np.float64)


def find_roots(coefficients):
    """
    Find roots of polynomial using NumPy.
    Args:
        coefficients: List of coefficients in descending order of degree
    Returns:
        Array of roots
    """
    return np.roots(coefficients)

def polynomial(coeffs, x):
   """Evaluate polynomial at x given coefficients in descending order"""
   return np.polyval(coeffs, x)


def main():
    max_tries = 600
    epsilon = 1e-6
    p = extractCoefficients("poly_coeff_alberth.txt")
    initGlobalMask(len(p) - 1)
    p_tag = derivative(p)
    start_time = time()
    aberth_roots = aberthEhrlich(p,p_tag, epsilon, max_tries)
    end_time = time()
    abert_time = end_time - start_time
    print("The operation took:", abert_time, "sec")

    aberth_roots = np.sort_complex(aberth_roots)
    # Example usage
    roots_  = find_roots(p)
    roots_ = np.sort_complex(roots_)
    print(np.c_[aberth_roots.reshape(-1,1),roots_.reshape(-1,1)])

if __name__ == "__main__":
    main()
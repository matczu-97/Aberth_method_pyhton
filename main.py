import numpy as np
from datetime import datetime
from time import time

p = []
polynomLen = 0
derivativeLen = 0
# Global mask initialization
mask = None
# Offsets array of complex zeros
initialized_zeros = None
# calculate p'
def derivative():
    global polynomLen
    return p[:-1] * np.arange(polynomLen - 1, 0, -1)

# calculate f(x)
def polyVal(f, x ,len):
    d = np.ones(len) * x
    d[0] = 1
    d = np.multiply.accumulate(d)
    return np.dot(d, f)

# calculate p(x) / q(x) both polynomials
def divide(p, q, x):
    if np.abs(x) <= 1:
        return (polyVal(p, x, polynomLen) / polyVal(q, x, derivativeLen))
    else:
        # Avoiding overflow case
        return x * (polyVal(p[::-1], 1 / x, polynomLen) / polyVal(q[::-1], 1 / x, derivativeLen))


# F = R*cos(theta) + i * R * sin(theha) Euler equation
def eulerEquation(R, theta):
    real_part = R * np.cos(theta)
    imag_part = R * np.sin(theta)
    return real_part + 1j * imag_part

# Initialize roots
def initRoots(p):
    global derivativeLen
    # number of roots is the number of coefficient of the derivative
    num_of_roots = derivativeLen
    # function to create Radius of the imaginary
    R = 1 + max(np.abs(p[1:])) / (np.abs(p[0]))
    roots = np.zeros(num_of_roots, dtype=complex)
    for i in range(num_of_roots):
        theta = (2 * np.pi * i) / polynomLen
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

# Initializing the zero complex for Offsets calculation W
def initStartingOffsets(n):
    global initialized_zeros
    initialized_zeros = np.zeros(n, dtype=complex)  # Set dtype to complex

# Calculate W equation from Aberth–Ehrlich and keeping the maximum
def calcOffset(p, p_tag, roots):
    Wmax = float('-inf')
    W = initialized_zeros
    for k, root in enumerate(roots):
        numerator = divide(p, p_tag, root)
        sigma = calculateSigma(roots, k)
        denominator = 1 - np.multiply(numerator, sigma)
        W[k] = np.divide(numerator, denominator)
        Wmax = max(Wmax, abs(W[k]))  # Use abs() for complex numbers
    return W, Wmax


# Aberth–Ehrlich  function to find roots.
def aberthEhrlich(p_tag, epsilon, max_tries):
    global p
    tries = 0
    roots = initRoots(p)
    while tries < max_tries:
        w, Wmax = calcOffset(p, p_tag, roots)
        # the alg stops when each offset is smaller than the defined epsilon
        if Wmax < epsilon:
            tries = max_tries
        # Updating roots with the offset -> SLIDE 6
        roots -= w
        tries += 1
    return roots


# extract coefficients  from file
def extractCoefficients(file_name):
    with open(file_name, 'r') as file:
        return np.array([float(line.strip()) for line in file], dtype=np.float64)


# printing the roots.
def printRoots(roots):
    formatted_roots = []
    for root in roots:
        if np.iscomplex(root):
            real_part = root.real
            imag_part = root.imag
            formatted_real = f"{real_part:.3f}"
            formatted_imag = f"{abs(imag_part):.3f}i"
            sign = "-" if imag_part < 0 else "+"
            formatted_roots.append(formatted_real + sign + formatted_imag)
        else:
            formatted_roots.append(f"{root:.3f}")
    print(formatted_roots)


def main():
    global p, polynomLen, derivativeLen
    max_tries = 800
    epsilon = 1e-4
    p = extractCoefficients("poly_coeff_alberth.txt")
    polynomLen = len(p)
    derivativeLen = polynomLen -1
    initGlobalMask(derivativeLen)
    initStartingOffsets(derivativeLen)
    p_tag = derivative()
    start_time = time()
    roots = aberthEhrlich(p_tag, epsilon, max_tries)
    end_time = time()
    abert_time = end_time - start_time
    printRoots(roots)
    print("The operation took:", abert_time, "sec")


if __name__ == "__main__":
    main()
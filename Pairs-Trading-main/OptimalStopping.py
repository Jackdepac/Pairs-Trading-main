# source for computation: https://arxiv.org/pdf/1411.5062.pdf
from math import sqrt, exp
import scipy.integrate as si
import scipy.optimize as so
import numpy as np

class OptimalStopping:
    '''
    Optimal Stopping Provides Functions for computing the Optimal Entry and Exit for our Pairs Portfolio

    Functions V and J are the functions used to calculate the Exit and Entry values, respectively
    '''
    def __init__(self, theta, mu, sigma, r, c):
        '''
        x - current portfolio value
        theta, mu, sigma - Ornstein-Uhlenbeck Coefficients
            (note we use self.theta for mean and self.mu for drift,
            while some sources use self.mu for mean and self.theta for drift)
        r - investor's subject discount rate
        c - cost of trading
        '''

        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.c = c

        self.b_star = self.b()
        self.F_of_b = self.F(self.b_star)

        self.d_star = self.d()

    def UpdateFields(self, theta=None, mu=None, sigma=None, r=None, c=None):
        '''
        Update our OU Coefficients
        '''    
        if theta is not None:
            self.theta = theta
        if mu is not None:
            self.mu = mu
        if sigma is not None:
            self.sigma = sigma
        if r is not None:
            self.r = r
        if c is not None:
            self.c = c

        self.b_star = self.b()
        self.F_of_b = self.F(self.b_star)
        
        self.d_star = self.d()

    def OptimalEntry(self):
        '''
        Optimal value to enter/buy the portfolio
        '''
        return self.d_star
    
    def OptimalExit(self):
        '''
        Optimal value to exit/liquidate the portfolio
        '''
        return self.b_star
    
    def V(self, x):
        # equation 4.2, solution of equation posed by 2.3

        if x < self.b_star:
            return (self.b_star - self.c) * self.F(x) / self.F_of_b
        else:
            return x - self.c

    def F(self, x):
        # equation 3.3
        def integrand(u):
            return u ** (self.r / self.mu - 1) * exp(sqrt(2 * self.mu / self.sigma ** 2) * (x - self.theta) * u - u ** 2 / 2)

        return si.quad(integrand, 0, np.inf)[0]

    def G(self, x):
        # equation 3.4
        def integrand(u):
            return u ** (self.r / self.mu - 1) * exp(sqrt(2 * self.mu / self.sigma ** 2) * (self.theta - x) * u - u ** 2 / 2)

        return si.quad(integrand, 0, np.inf)[0]

    def b(self):
        # estimates b* using equation 4.3

        def func(b):
            return self.F(b) - (b - self.c) * self.Prime(self.F, b)

        # finds the root of function between the interval [0, 1]
        return so.brentq(func, 0, 1)

    def d(self):
        # estimates d* using equation 4.11

        def func(d):
            return (self.G(d) * (self.Prime(self.V, d) - 1)) - (self.Prime(self.G, d) * (self.V(d) - d - self.c))

        # finds the root of function between the interval [0, 51
        return so.brentq(func, 0, 1)

    def Prime(self, f, x, h=1e-4):
        # given f, estimates f'(x) using the difference quotient forself.mula 
        # WARNING: LOWER h VALUES CAN LEAD TO WEIRD RESULTS
        return (f(x + h) - f(x)) / h
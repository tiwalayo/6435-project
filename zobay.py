from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.linalg import sqrtm

PI = 3.1415926535897932384626
E = 2.718281828459045

omega = lambda k: 2 * PI**((k-1)/2) / gamma((k-1)/2)
c = lambda k: 0.39894228040143267793**k

class VGMM:
    def __init__(self, k=2, d=1):
        """We have a few parameters:
        sigma_0 = base sigma
        lambda_i (i \in range(1, k+1)) = sigma multiple
        w_i (i \in range(1, k+1)) = weight
        mu_i = mean
        """

        self.sigma_0 = np.eye(d)
        self.lambdas = np.random.rand((k, 1))
        self.ws = np.random.rand((k, 1))
        self.mus = np.random.rand((k, d))

        self.k = k
        self.d = d

        self.eta = 0.001 # learning rate
    
   def setTarget(self, dist):
       # idea: compute sigma_0 based on ELBO
       self.targetpdf = dist
       self.logp = lambda x: np.log(dist(x))

    def coordinateDescent(self, rounds=10, entropyRounds=30, energyRounds=30):
        # TODO: allow "change < epsilon" flag
        maxrounds = rounds
        round = 0
        while True:
            if round == maxrounds:
                break
            self.entropyDescent(rounds=entropyRounds)
            self.energyDescent(rounds=energyRounds)
            round += 1

    def entropyDescent(self, rounds=30):
        for round in range(rounds):
            # update wrt individual entropies
            wGrads = -1/self.ws + 1/2 * np.log(np.linalg.det(2*PI*E*self.lambdas * self.sigma_0))
            lambdaGrads = self.ws * self.d/self.lambdas

            self.ws -= wGrads * self.eta
            self.lambdas -= lambdaGrads * self.eta

            # update wrt pairwise entropies
            for i in range(self.k):
                for j in range(j):
                    w1, w2 = self.ws[i], self.ws[j]
                    lmbda = self.lambdas[j]
                    siginv = np.linalg.inv(self.sigma_0)
                    mu = self.mus[i] - self.mus[j]
                    r = (mu @ siginv @ mu.T)**0.5
                    softr = U @ mu.T
                    U = sqrtm(siginv)
                    Uinv = np.linalg.inv(U)                

                    # convenience functions
                    bigexp = lambda z, p: np.exp(-1/(2 * lmbda**2) * (z-r)**2 + 0.5*z**2 - 0.5 * (lmbda**(-2)-1)*p**2)
                    ghqintbase = lambda z, p: 2 * omega(self.k-1) * c(self.k) * (1.414 * p)**(k-2) * 1/(1 + w2/(w1 * lmbda**self.k) * np.exp(-1/(2 * lmbda**2) * ((1.414*z)-r)**2 + 0.5*(1.414*z)**2 - 0.5 * (lmbda**(-2)-1)*(1.414*p)**2))
                    ghqintbase2 = lambda z, p: 2 * omega(self.k-1) * c(self.k) * (1.414 * p)**(k-2) * np.log(1 + w2/(w1 * lmbda**self.k) * np.exp(-1/(2 * lmbda**2) * ((1.414*z)-r)**2 + 0.5*(1.414*z)**2 - 0.5 * (lmbda**(-2)-1)*(1.414*p)**2))

                    # gradients with respect to the GHQ integral  
                    lambdafn = lambda z, p, w1, w2: ghqintbase(z,p) * (-w2/w1 * self.k * lmbda**(-k-1) * bigexp(z,p) + w2/(w1*lmbda**k) * bigexp(z,p) * (lmbda**(-3)*((z-r)**2 - p**2 )))
                    w1fn = lambda z, p, w1, w2: ghqintbase(z,p) * bigexp(z,p) * w2/lmbda**k * -1/w1**2
                    w2fn = lambda z, p, w1, w2: ghqintbase(z,p) * bigexp(z,p) * 1/(w1 * lmbda**k)
                    rfn = lambda z, p, w1, w2: ghqintbase(z,p) * w2/(w1*lmbda**k) * bigexp(z,p) * lmbda**(-2) * (z-r)

                    # updates
                    self.lambdas[j] -= (-self.ws[i] * self.ghq(partial(lambdafn, w1=w1, w2=w2)) - self.ws[j] * self.ghq(partial(lambdafn, w1=w2, w2=w1))) * self.eta
                    self.ws[i] -= (-1 * self.ghq(ghqintbase2) * w1 * self.ghq(partial(w1fn, w1=w1, w2=w2)) - w2 * self.ghq(partial(w2fn, w1=w2, w2=w1))) * self.eta
                    self.ws[j] -= (-1 * self.ghq(ghqintbase2) * w2 * self.ghq(partial(w2fn, w1=w1, w2=w2)) - w1 * self.ghq(partial(w1fn, w1=w2, w2=w1))) * self.eta
                    softr -= (-w1 * self.ghq(partial(rfn, w1=w1, w2=w2)) - w2 * self.ghq(partial(rfn, w1=w2, w2=w1))) * self.eta
                    muchange = -1 * (Uinv @ softr - mu)
                    
                    # it is currently unclear to me how to optimize two variables given the optimization of their difference; here's a hack i thought up:
                    factor = np.random.uniform(-1, 1)
                    self.mus[i] += factor * muchange
                    self.mus[j] += (1+factor) * muchange

    def energyDescent(self, rounds=30):
        for round in range(rounds):
            for i in range(self.mus.shape[0]):
                mu = self.mus[i]
                siginv = np.linalg.inv(self.sigma_0 * self.lambdas[i])
                sig = self.sigma_0 * self.lambdas[i]
                mufn = lambda x: self.logp(x) * self.ws[i] * (2*PI)**(-k/2) * (np.linalg.det(self.sigma_0 * self.lambdas[i])**(-0.5) * np.exp(-0.5 * (x-mu)*siginv*(x-mu).T) * siginv * (x-mu))
                lambdafn = lambda x: self.logp(x) * self.ws[i] * (2*PI)**(-k/2) * (np.linalg.det(self.sigma_0 * self.lambdas[i])**(-0.5) * np.exp(-0.5 * (x-mu)*siginv*(x-mu).T) * 0.5 * siginv * (x-mu) * (x-mu).T * siginv * sig)
                # the above line is problematic. this should yield a diagonal matrix of the form c*id but that's not what we get, and hours of redoing the math doesn't give me a different answer

                self.mus[i] -= self.ghqX(mufn) * self.eta
                self.lambdas[i] -= self.ghqX(lambdafn) * self.eta


    # Gauss-Hermite quadrature to compute normal form gradient integrals in cylindrical (z, p) coordinates
    def ghq(self, fn):
        zsamplepts, zweights = np.polynomial.hermite.hermgauss(20)
        psamplepts, pweights = np.polynomial.hermite.hermgauss(20)
        
        weightOP = (zweights * np.expand_dims(pweights[:-10], axis=1))

        ghqsamples = np.transpose([numpy.tile(zsamplepts, len(psamplepts[:-10])), numpy.repeat(psamplepts[:-10], len(zsamplepts))])
        ghqpts = np.array([fn(z,p) for z,p in ghqsamples]).reshape(20,10)

        return (weightOP * ghqpts).sum()
    
    # Gauss-Hermite quadrature to compute ordinary exponent-type integrals
    def ghqX(self, fn):
        xsamplepts, xweights = np.polynomial.hermite.hermgauss(20)
        
        ghqpts = np.array([fn(x) for x in xsamplepts]).reshape(20)

        return (xweights * ghqpts).sum()
    
    # print results
    def printout(self, plot=False):
        print("weights", self.ws)
        print("mus", self.mus)
        print("sigma_0", self.sigma_0)
        print("lambdas", self.lambdas)

        if plot:
            if self.d > 1:
                raise ValueError("Dimensionality is too large for a plot of the resultant mixture.")

            clustersamples = np.random.multinomial(300, self.ws, size=1)

            data = np.array([])
            for i in range(len(clustersamples)):
                data = np.append(data, np.random.multivariate_normal(self.mus[i], self.lambdas[i] * self.sigma_0, clustersamples[i]))

            plt.hist(data, density=True)
            plt.show()
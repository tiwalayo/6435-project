from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.linalg import sqrtm
import scipy.stats

PI = 3.1415926535897932384626
E = 2.718281828459045
EPSILON = 0.000001

omega = lambda k: 2 * PI**((k-1)/2) / gamma((k-1)/2)
c = lambda k: 0.39894228040143267793**k

# utils
import numpy as np
from scipy.stats import multivariate_normal

def multidet(M):
  # Determinant of multidimensional matrix
  dm = np.zeros(M.shape[0])
  for _ in range(len(dm)):
    dm[_] = np.linalg.det(M[_])
  return dm[np.newaxis].T

# 90% sure this can be done using broadcasting:
def multimult(A, B):
  # "Broadcast" multiplication of matrices with vector
  dm = np.zeros(A.shape + B.shape)
  for _ in range(len(dm)):
    dm[_] = A[_] * B
  return dm

def gaussianExpectation(mu, cov, fn, n_samples=30):
  samples = np.random.multivariate_normal(mu, cov, n_samples)
  return np.mean(np.apply_along_axis(fn, 1, samples), axis=0)

def mvNormpdf(x, mu, cov):
  k = mu.shape[0]
  return 1/((2*np.pi)**(k/2)*(np.linalg.det(sigma)**0.5))* np.exp(-0.5 * ((x-mu).T @ (2 * np.eye(2)) @ (x-mu)).sum())

def logRatio(mu1, mu2, sig1, sig2, w1, w2):
  return np.log(1 + (w2 * mvNormpdf(x, mu2, sig2))/(w1 * mvNormpdf(x, mu1, sig1)))

def pairwiseObjective(mu1, mu2, sig1, sig2, w1, w2):
  return -w1 * gaussianExpectation() - w2 * gaussianExpectation()
  
class VGMM:
    def __init__(self, k=1, d=1):
        """We have a few parameters:
        sigma_0 = base sigma
        lambda_i (i \in range(1, k+1)) = sigma multiple
        w_i (i \in range(1, k+1)) = weight
        mu_i = mean
        """

        self.sigma_0 = np.eye(d)
        self.lambdas = np.random.rand(k, 1)
        self.ws = np.random.rand(k, 1)
        self.ws = self.ws/self.ws.sum()
        self.mus = np.random.rand(k, d)

        self.k = k
        self.d = d

        self.eta = 0.006 # learning rate

    def setTarget(self, dist):
       # idea: compute sigma_0 based on ELBO
       dist1 = scipy.stats.norm(-1, 0.5).pdf
       dist2 = scipy.stats.norm(2, 1).pdf
       self.targetpdf = lambda x: 0 * dist1(x) + 1 * dist2(x)
       self.logp = lambda x: np.log(self.targetpdf(x)) if np.log(self.targetpdf(x)) > -100 else -100

    def coordinateDescent(self, rounds=3, entropyRounds=1, energyRounds=1):
        # TODO: allow "change < epsilon" flag
        maxrounds = rounds
        round = 0
        while True:
            if round == maxrounds:
                break
            
            for i in range(self.lambdas.shape[0]):
                self.entropyDescent(rounds=entropyRounds, param=("lambda", i))
                self.energyDescent(rounds=energyRounds, param=("lambda", i))
            for i in range(self.ws.shape[0]):
                self.entropyDescent(rounds=entropyRounds, param=("w", i))
                self.energyDescent(rounds=energyRounds, param=("w", i))
            for i in range(self.mus.shape[0]):
                self.entropyDescent(rounds=entropyRounds, param=("mu", i))
                self.energyDescent(rounds=energyRounds, param=("mu", i))

            round += 1

    def entropyDescent(self, rounds=30, param=None):
        # maximize entropy wrt params 
        self.whist = []
        self.lambdahist = []
        for round in range(rounds):
            # update wrt individual entropies

            # det of arrays within array, see https://stackoverflow.com/questions/13393733/determinant-of-multidimensional-array for optimization options
            if round % 10 == 0:
              # print(round, multimult(self.lambdas, self.sigma_0))
              # print(round, self.mus)
              pass
            wGrads = -1/(self.ws+EPSILON) + 1/2 * np.log(multidet(2*PI*E*multimult(self.lambdas**2, self.sigma_0)))
            lambdaGrads = self.ws * self.d/self.lambdas

            var, ind = param
            # only update one variable at a time
            if var == "w":
                mask = np.zeros(self.ws.shape)
                mask[ind] = np.ones(self.ws[ind].shape)
                self.ws += (wGrads * self.eta) * mask
            
            if var == "lambda":
                mask = np.zeros(self.lambdas.shape)
                mask[ind] = np.ones(self.lambdas[ind].shape)
                self.lambdas += (lambdaGrads * self.eta) * mask

            self.whist.append((len(self.whist), self.ws[0][0]))
            self.lambdahist.append((len(self.lambdahist), self.lambdas[0][0]))

            # update wrt pairwise entropies
            for j in range(self.k):
                if j != ind:
                    w1, w2 = self.ws[ind], self.ws[j]
                    sig1, sig2 = self.lambdas[ind] * self.sigma_0, self.lambdas[j] * self.sigma_0
                    mu1, mu2 = self.mus[ind], self.mus[j]

                    if var == "mu":
                        self.mus[ind] = scipy.optimize.minimize(partial(component_objective,f=f,n=n,ip_fg=ip_fg,weights=lambdas[n-2,:n-1],gs=gs),h_opt,method='Powell',bounds=((None,None),(0,None))).x
                    r = (mu @ siginv @ mu.T)**0.5
                    U = sqrtm(siginv)
                    Uinv = np.linalg.inv(U)
                    
                    softr = U @ mu.T

                    k = self.k

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
                    self.lambdas[j] += (-self.ws[i] * self.ghq(partial(lambdafn, w1=w1, w2=w2)) - self.ws[j] * self.ghq(partial(lambdafn, w1=w2, w2=w1))) * self.eta
                    self.ws[i] += (-1 * self.ghq(ghqintbase2) * w1 * self.ghq(partial(w1fn, w1=w1, w2=w2)) - w2 * self.ghq(partial(w2fn, w1=w2, w2=w1))) * self.eta
                    self.ws[j] += (-1 * self.ghq(ghqintbase2) * w2 * self.ghq(partial(w2fn, w1=w1, w2=w2)) - w1 * self.ghq(partial(w1fn, w1=w2, w2=w1))) * self.eta
                    softr += (-w1 * self.ghq(partial(rfn, w1=w1, w2=w2)) - w2 * self.ghq(partial(rfn, w1=w2, w2=w1))) * self.eta
                    muchange = -1 * (Uinv @ softr - mu)
                    
                    # it is currently unclear to me how to optimize two variables given the optimization of their difference; here's a hack i thought up:
                    factor = np.random.uniform(-1, 1)
                    self.mus[i] += factor * muchange
                    self.mus[j] += (1+factor) * muchange
                    print(muchange)

    def energyDescent(self, rounds=30):
        # minimize energy $-\int q \log p$ wrt params
        # as it stands, every round takes about 0.025 sec (horrible!); not sure i want to decrease the sample count in gaussExpectation tho
        self.lambdahist = []
        self.muhist = []
        #print("siginv", siginv)
        for round in range(rounds):
            for i in range(self.mus.shape[0]):
                mu = self.mus[i].T
                siginv = np.linalg.inv(self.sigma_0 * self.lambdas[i]**2)
                sig = self.sigma_0 * self.lambdas[i] ** 2

                mufn = lambda x: self.logp(x) * siginv @ (x - mu[np.newaxis])
                lambdafn = lambda x: self.logp(x) * -0.5 * (siginv - siginv @ (x - mu[np.newaxis]) @ (x - mu[np.newaxis]).T @ siginv) @ (2 * self.lambdas[i] * self.sigma_0)

                self.mus[i] -= -1 * np.squeeze(self.ws[i] * gaussianExpectation(mu, sig, mufn) * self.eta, axis=-1)
                self.lambdas[i] -= -1 * self.ws[i] * gaussianExpectation(mu, sig, lambdafn)[0][0] * self.eta # expectation should be diagonal matrix

                self.lambdahist.append((len(self.lambdahist), self.lambdas[0][0]))
                self.muhist.append((len(self.muhist), self.mus[0][0]))

    # Gauss-Hermite quadrature to compute normal form gradient integrals in cylindrical (z, p) coordinates
    def ghq(self, fn):
        zsamplepts, zweights = np.polynomial.hermite.hermgauss(20)
        psamplepts, pweights = np.polynomial.hermite.hermgauss(20)
        
        # cartesianish product of z & p weights
        weightOP = (zweights * np.expand_dims(pweights[:-10], axis=1)) # only integrate over positive p

        # cartesian product of z & p samples
        ghqsamples = np.transpose([np.tile(zsamplepts, len(psamplepts[:-10])), np.repeat(psamplepts[:-10], len(zsamplepts))])
        ghqpts = np.array([fn(z,p) for z,p in ghqsamples]).reshape(10,20)

        return (weightOP * ghqpts).sum()
        
    # Gauss-Hermite quadrature to compute ordinary exponent-type integrals
    def ghqX(self, fn):
        xsamplepts, xweights = np.polynomial.hermite.hermgauss(20)
        
        ghqpts = np.array([fn(x) for x in xsamplepts]).reshape(20)

        return (xweights * ghqpts).sum()

    # Compute integrals of the type \int q_1 \log(1 + (w2 * q2)/(w1 * q*1)) dx, i.e. I-type integrals
    def iTypeInt(self, mu1, mu2, sig1, sig2, w1, w2):
      np.log(1+(w2*x)/(w1*y))
    
    # print results
    def printout(self, plot=False):
        print("weights", self.ws)
        print("mus", self.mus)
        print("sigma_0", self.sigma_0)
        print("lambdas", self.lambdas)

        if plot:
            if self.d > 2:
                raise ValueError("Dimensionality is too large for a plot of the resultant mixture.")

            clustersamples = np.random.multinomial(300, self.ws, size=1)

            data = np.array([])
            for i in range(len(clustersamples)):
                data = np.append(data, np.random.multivariate_normal(self.mus[i], self.lambdas[i] * self.sigma_0, clustersamples[i]))

            plt.hist(data, density=True)
            plt.show()
  
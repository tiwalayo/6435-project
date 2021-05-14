from functools import partial
from re import A

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, softmax
from scipy.linalg import sqrtm
import scipy.stats
from tqdm import tqdm

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

def sigmabroadcast(sig, samples):
  # "Broadcast" multiplication of matrices with vector
  dm = np.zeros((samples.shape[0], sig.shape[0], sig.shape[0]))
  for _ in range(dm.shape[0]):
    dm[_] = sig @ samples[0][np.newaxis].T
  return dm

def gaussianExpectation(mu, cov, fn, n_samples=1000, presamples=None):
  if type(presamples) == type(None):
    samples = np.random.multivariate_normal(mu, cov, n_samples)
  else:
    samples = presamples
  return np.mean(np.apply_along_axis(fn, 1, samples), axis=0)

def mvNormpdf(x, mu, cov):
  k = mu.shape[0]
  #print(x.shape, mu.shape, cov.shape)
  return 1/((2*np.pi)**(k/2)*(np.linalg.det(cov)**0.5))* np.exp(-0.5 * ((x-mu).T @ (np.linalg.pinv(cov)) @ (x-mu)).sum())

def logRatio(mu1, mu2, sig1, sig2, w1, w2):
  return np.log(1 + (w2 * mvNormpdf(x, mu2, sig2))/(w1 * mvNormpdf(x, mu1, sig1)))

def pairwiseObjective(mu1, mu2, sig1, sig2, w1, w2):
  return -w1 * gaussianExpectation() - w2 * gaussianExpectation()

class Param:
  def __init__(self):
    self.m_prev = 0
    self.v_prev = 0
  
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
        self.Ws = np.random.rand(k, 1)
        self.ws = softmax(self.Ws)
        self.mus = np.random.rand(k, d)

        self.k = k
        self.d = d

        self.eta = 0.01 # learning rate

        #adam
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 10**(-8)
        self.m_prev = 0
        self.v_prev = 0
        self.adamparams = {}

        self.n_samples = 1000
        self.cgd = False

        self.whist = []
        self.lambdahist = [[] for i in range(k)]
        self.lambdaghist = [[] for i in range(k)]
        self.mughist = [[] for i in range(k)]
        self.muhist = [[] for i in range(k)]
        self.h = []

    def get_grad(self, grad, params=None):
      return self.adam(grad, params=params)
    
    def adam(self, grad, params):
      if params['id'] not in self.adamparams:
        self.adamparams[params['id']] = Param()
      s = self.adamparams[params['id']]

      s.m_current = self.beta_1 * s.m_prev + (1 - self.beta_1) * grad
      s.v_current = self.beta_2 * s.v_prev + (1 - self.beta_2) * grad**2
      s.m_norm = s.m_current/(1 - self.beta_1**(params['t']+1))
      s.v_norm = s.v_current/(1 - self.beta_2**(params['t']+1))

      s.m_prev = s.m_current * 1.0
      s.v_prev = s.v_current * 1.0

      if params['id'] == 'lambda':
        self.h.append(self.eta * s.m_norm/((s.v_norm**0.5) + self.epsilon))
      return self.eta * s.m_norm/((s.v_norm**0.5) + self.epsilon)

    def setTarget(self, dist):
       # idea: compute sigma_0 based on ELBO
       dist1 = scipy.stats.norm(-1, 0.5).pdf
       dist2 = scipy.stats.norm(1, 2).pdf
       self.targetpdf = lambda x: 0 * dist1(x) + 1 * dist2(x)
       self.logp = lambda x: np.log(self.targetpdf(x))
      #  self.logp = lambda x: np.log(self.targetpdf(x)) if np.log(self.targetpdf(x)) > -100 else -100

    def coordinateDescent(self, rounds=500, entropyRounds=1, energyRounds=1):
        # TODO: allow "change < epsilon" flag
        maxrounds = rounds
        round = 0
        
        for round in tqdm(range(maxrounds)):
            self.round = round
            for i in range(self.ws.shape[0]):
                self.entropyDescent(rounds=entropyRounds, param=("w", i))
                self.energyDescent(rounds=energyRounds, param=("w", i))
                self.ws = softmax(self.Ws)
            for i in range(self.lambdas.shape[0]):
                self.entropyDescent(rounds=entropyRounds, param=("lambda", i))
                self.energyDescent(rounds=energyRounds, param=("lambda", i))
                pass
            for i in range(self.mus.shape[0]):
                self.entropyDescent(rounds=entropyRounds, param=("mu", i))
                self.energyDescent(rounds=energyRounds, param=("mu", i))

    def entropyDescent(self, rounds=1, param=None):
        
        # maximize entropy wrt params 
        self.ws = softmax(self.Ws)
        for round in range(rounds):
            # update wrt individual entropies

            # det of arrays within array, see https://stackoverflow.com/questions/13393733/determinant-of-multidimensional-array for optimization options
            if round % 10 == 0:
              # print(round, multimult(self.lambdas, self.sigma_0))
              # print(round, self.mus)
              pass
            wGrads = -(np.log(self.ws)+1) + 1/2 * np.log(multidet(2*PI*E*multimult(self.lambdas**2, self.sigma_0))) * self.ws * (1 - self.ws)
            lambdaGrads = self.ws * self.d/(self.lambdas + EPSILON)

            var, ind = param
            # only update one variable at a time
            if var == "w" or not self.cgd:
                # mask = np.zeros(self.Ws.shape)
                # mask[ind] = np.ones(self.Ws[ind].shape)
                self.Ws += self.get_grad(wGrads, params={'t':self.round, 'id':'node-W'})
            
            if var == "lambda" or not self.cgd:
                # mask = np.zeros(self.lambdas.shape)
                # mask[ind] = np.ones(self.lambdas[ind].shape)
                self.lambdas += self.get_grad(lambdaGrads, params={'t':self.round, 'id':'node-lambdas'})

            # self.whist.append((len(self.whist), self.ws[0][0]))
            self.muhist[ind].append(1.0 * self.mus[ind])
            self.lambdahist[ind].append(1.0 * self.lambdas[ind])
            self.lambdaghist[ind].append(1.0 * lambdaGrads[ind])

            # update wrt pairwise entropies
            Q1SAMPLES = np.random.multivariate_normal(self.mus[ind], self.lambdas[ind]**2 * self.sigma_0, self.n_samples)
            self.Q1SAMPLES = Q1SAMPLES
            for j in range(self.k):
                if j != ind:
                    w1, w2 = self.ws[ind], self.ws[j]
                    sig1, sig2 = self.lambdas[ind]**2 * self.sigma_0, self.lambdas[j]**2 * self.sigma_0
                    sig1inv = np.linalg.inv(self.sigma_0 * self.lambdas[ind]**2)
                    sig2inv = np.linalg.inv(self.sigma_0 * self.lambdas[j]**2)
                    mu1, mu2 = self.mus[ind], self.mus[j]

                    # cache results
                    pdf1 = scipy.stats.multivariate_normal.pdf(Q1SAMPLES, mean=mu1, cov=sig1)#mvNormpdf(Q1SAMPLES, mu1, sig1)
                    pdf2 = scipy.stats.multivariate_normal.pdf(Q1SAMPLES, mean=mu2, cov=sig2)#mvNormpdf(Q1SAMPLES, mu2, sig2)

                    if var == "w" or not self.cgd:
                      #wfn1 = lambda x: np.log(1+(w2*pdf2/(w1 * pdf1))) + 1/(1+(w2*pdf2/(EPSILON + w1 * pdf1))) * (w2 * pdf2/(EPSILON + pdf1)) * (-1/(EPSILON + w1))
                      # evaluate wfn1 at Q1SAMPLES
                      wfn1_exp = np.log(1+(w2*pdf2/(w1 * pdf1))) + 1/(1+(w2*pdf2/(EPSILON + w1 * pdf1))) * (w2 * pdf2/(EPSILON + pdf1)) * (-1/(EPSILON + w1))
                      wfn1_exp = np.mean(wfn1_exp)

                      #wfn2 = lambda x: 1/(1 + (w1 * pdf1)/(w2 * pdf2)) * 1/(EPSILON + w2)
                      # evaluate wfn2 at !sSAMPLES
                      wfn2_exp = 1/(1 + (w1 * pdf1)/(w2 * pdf2)) * 1/(EPSILON + w2)
                      wfn2_exp = np.mean(wfn2_exp)

                      #self.Ws[ind] += self.get_grad(-gaussianExpectation(None, None, wfn1, presamples=Q1SAMPLES) - w2 * gaussianExpectation(None, None, wfn2, presamples=Q1SAMPLES) * w1 * (1 - w1), params={'t':self.round, 'id':'pair-'+str(ind)+','+str(j)+'-W'})

                      # use expectations to calculate gradient
                      self.Ws[ind] += self.get_grad(-wfn1_exp - w2 * wfn2_exp * w1 * (1 - w1), params={'t':self.round, 'id':'pair-'+str(ind)+','+str(j)+'-W'})

                    if var == "mu" or not self.cgd:
                      #omega1 = lambda x: 1/(1+(w2*pdf2/(EPSILON + w1 * pdf1))) * (w2 * pdf2/(EPSILON + pdf1)) * (-1/(EPSILON + pdf1**2)) * pdf1 * sig1inv @ (x - mu1[np.newaxis])
                      omega1 = 1/(1+(w2*pdf2/(EPSILON + w1 * pdf1))) * (w2 * pdf2/(EPSILON + pdf1)) * (-1/(EPSILON + pdf1**2)) * pdf1 * -sig1inv @ (Q1SAMPLES - mu1[np.newaxis])

                      #mufn1 = lambda x: sig1inv @ (x - mu1[np.newaxis]) * np.log(1+(w2*pdf2/(w1 * pdf1))) + omega1(x)
                      mufn1_exp = -sig1inv @ (Q1SAMPLES - mu1[np.newaxis]).T * np.log(1+(w2*pdf2/(w1 * pdf1))) + omega1
                      mufn1_exp = np.mean(mufn1_exp)

                      #mufn2 = lambda x: pdf2 * 1/(1 + (w1 * pdf1)/(w2 * pdf2)) * w1/(EPSILON + w2 * pdf2) * sig1inv @ (x - mu1[np.newaxis])
                      postprefix = -sig1inv @ (Q1SAMPLES - mu1[np.newaxis]).T
                      mufn2_exp = pdf2 * 1/(1 + (w1 * pdf1)/(w2 * pdf2)) * w1/(EPSILON + w2 * pdf2) * postprefix
                      mufn2_exp = np.mean(mufn2_exp)

                      self.mus[ind] += self.get_grad(np.squeeze((-w1 * mufn1_exp - w2 * mufn2_exp), axis=-1), params={'t':self.round, 'id':'pair-'+str(ind)+','+str(j)+'-mu'})
                      
                    if var == "lambda" or not self.cgd:
                      # omega1 = lambda x: 1/(1+(w2*pdf2/(EPSILON + w1 * pdf1))) * (w2 * pdf2/(EPSILON + pdf1)) * (-1/(EPSILON + pdf1**2)) * pdf1 * (sig1inv - sig1inv @ (x - mu1[np.newaxis]) @ (x - mu1[np.newaxis]).T @ sig1inv) @ (self.lambdas[ind] * self.sigma_0)
                      # print(sig1inv.shape, (Q1SAMPLES - mu1[np.newaxis]).shape, (Q1SAMPLES - mu1[np.newaxis]).T.shape, sig1inv)
                      postprefix = (sig1inv - sig1inv @ (Q1SAMPLES - mu1[np.newaxis]).T @ (Q1SAMPLES - mu1[np.newaxis]) @ sig1inv)
                      postpostprefix = (self.lambdas[ind] * self.sigma_0)
                      omega1 = 1/(1+(w2*pdf2/(EPSILON + w1 * pdf1))) * (w2 * pdf2/(EPSILON + pdf1)) * (-1/(EPSILON + pdf1**2)) * pdf1 * postprefix * postpostprefix

                      #lambdafn1 = lambda x: (sig1inv - sig1inv @ (x - mu1[np.newaxis]) @ (x - mu1[np.newaxis]).T @ sig1inv) @ (self.lambdas[ind] * self.sigma_0) * np.log(1+(w2*pdf2/(w1 * pdf1))) + omega1(x)
                      lambdafn1_exp = (sig1inv - sig1inv @ (Q1SAMPLES - mu1[np.newaxis]).T @ (Q1SAMPLES - mu1[np.newaxis]) @ sig1inv) * (self.lambdas[ind] * self.sigma_0) * np.log(1+(w2*pdf2/(w1 * pdf1))) + omega1
                      lambdafn1_exp = np.mean(lambdafn1_exp)

                      # lambdafn2 = lambda x: pdf2 * 1/(1 + (w1 * pdf1)/(w2 * pdf2)) * w1/(EPSILON + w2 * pdf2) * (sig1inv - sig1inv @ (x - mu1[np.newaxis]) @ (x - mu1[np.newaxis]).T @ sig1inv) @ (self.lambdas[ind] * self.sigma_0)
                      lambdafn2_exp = pdf2 * 1/(1 + (w1 * pdf1)/(w2 * pdf2)) * w1/(EPSILON + w2 * pdf2) * (sig1inv - sig1inv @ (Q1SAMPLES - mu1[np.newaxis]).T @ (Q1SAMPLES - mu1[np.newaxis]) @ sig1inv) * (self.lambdas[ind] * self.sigma_0)
                      lambdafn2_exp = np.mean(lambdafn2_exp)

                      self.lambdas[ind] += self.get_grad(-w1 * lambdafn1_exp - w2 * lambdafn2_exp, params={'t':self.round, 'id':'pair-'+str(ind)+','+str(j)+'-lambda'})

    def energyDescent(self, rounds=1, param="dummy variable"):
        # minimize energy $-\int q \log p$ wrt params
        self.ws = softmax(self.Ws)

        for round in range(rounds):
            for i in range(self.mus.shape[0]):
                mu = self.mus[i].T
                siginv = np.linalg.inv(self.sigma_0 * self.lambdas[i]**2)
                sig = self.sigma_0 * self.lambdas[i]** 2
                w = self.ws[i]

                Q1SAMPLES = np.random.multivariate_normal(mu, sig, self.n_samples)

                # mufn = lambda x: self.logp(x) * siginv @ (x - mu[np.newaxis])
                mufn_exp = self.logp(Q1SAMPLES) * sigmabroadcast(-siginv, (Q1SAMPLES - mu[np.newaxis]))
                mufn_exp = np.mean(mufn_exp)
                

                # lambdafn = lambda x: self.logp(x) * -0.5 * (siginv - siginv @ (x - mu[np.newaxis]) @ (x - mu[np.newaxis]).T @ siginv) @ (2 * self.lambdas[i] * self.sigma_0)
                lambdafn_exp = self.logp(Q1SAMPLES) * -0.5 * (siginv - siginv @ (Q1SAMPLES - mu[np.newaxis]).T @ (Q1SAMPLES - mu[np.newaxis]) @ siginv) * (2 * self.lambdas[i] * self.sigma_0)
                lambdafn_exp = np.mean(lambdafn_exp)

                logp_exp = self.logp(Q1SAMPLES)
                logp_exp = np.mean(logp_exp)

                self.mus[i] -= self.get_grad(np.squeeze(-self.ws[i] * mufn_exp, axis=-1), params={'t':self.round, 'id':'energy-mu-'+str(i)})*1.5
                self.lambdas[i] -= self.get_grad(-1 * self.ws[i] * lambdafn_exp, params={'t':self.round, 'id':'energy-lambda-'+str(i)}) # expectation should be diagonal matrix
                self.Ws -= self.get_grad(-1 * logp_exp * w * (1 - w), params={'t':self.round, 'id':'energy-W'+str(i)})

                # self.lambdahist.append((len(self.lambdahist), self.lambdas[0][0]))
                self.muhist[i].append(1.0 * self.mus[i])
                self.lambdahist[i].append(1.0 * self.lambdas[i])
                self.mughist[i].append(1.0 * np.squeeze(-self.ws[i] * mufn_exp, axis=-1))
    
    # print results
    def printout(self, plot=False, show_target=True):
        print("weights", self.ws)
        print("mus", self.mus)
        print("sigma_0", self.sigma_0)
        print("lambdas", self.lambdas)

        if plot:
            if self.d > 2:
                raise ValueError("Dimensionality is too large for a plot of the resultant mixture.")

            self.ws = softmax(self.Ws)

            clustersamples = np.random.multinomial(10000, (self.ws).flatten(), size=1)[0]

            data = np.array([])
            for i in range(len(clustersamples)):
                data = np.append(data, np.random.multivariate_normal(self.mus[i], self.lambdas[i]**2 * self.sigma_0, clustersamples[i]))

            plt.hist(data, density=True, bins=30)
            if show_target==True:
              plt.plot(list(np.arange(-5,5,0.05)), [self.targetpdf(x) for x in list(np.arange(-5,5,0.05))])
            plt.show()
  

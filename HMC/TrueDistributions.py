import tensorflow as tf
import numpy as np
class MVN:
  def __init__(self, mu:tf.Tensor, cov:tf.Tensor):
    self.mu = mu
    self.cov = cov
    self.dim = len(mu)

  def logPdf(self, x:np.ndarray):
    return -0.5 * (tf.math.log(tf.linalg.det(self.cov)) + (x-self.mu).T @ tf.linalg.inv(self.cov) @ (x-self.mu))
  def drawSample(self, n:int):
    return np.random.multivariate_normal(self.mu, self.cov, n)
  
  def getMean(self):
    return self.mu
  
  def getCovariance(self):
    return self.cov
  

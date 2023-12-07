import tensorflow as tf
import tensorflow_probability as tfp
from HMC.TrueDistributions import *

class HamiltonianChainMaker:
    def __init__(self, dist: MVN,n=10**4,step_size=1.0) -> None:
        tf.experimental.numpy.experimental_enable_numpy_behavior()
        self.output = None
        self.acceptance_rate = None
        self.dist = dist
        self.n = n
        self.step_size = step_size
        self.createSamples()


    def unnormalized_log_prob(self,x):
        return self.dist.logPdf(x)

    def createSamples(self):    
        # Initialize the HMC transition kernel.
        num_burnin_steps = self.n//20
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.unnormalized_log_prob,
                num_leapfrog_steps=5,
                step_size=self.step_size),
            num_adaptation_steps=int(num_burnin_steps * 0.8))

        # Run the chain (with burn-in).
        @tf.function
        def run_chain(n):
            # Run the chain (with burn-in).
            samples, is_accepted = tfp.mcmc.sample_chain(
                num_results=n,
                num_burnin_steps=num_burnin_steps,
                current_state=[tf.cast([1,0], dtype=tf.float32)],
                kernel=adaptive_hmc,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

            sample_mean = tf.reduce_mean(samples)
            sample_stddev = tf.math.reduce_std(samples)
            # is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))

            return sample_mean, sample_stddev, is_accepted, samples

        sample_mean, sample_stddev, is_accepted, samples = run_chain(self.n)
        self.output = samples[0].numpy()
        self.is_accepted = is_accepted.numpy()
        self.acceptance_rate = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
    
    def getOutput(self):
        return self.output[self.is_accepted]
    
    def getAcceptanceRate(self):
        return self.acceptance_rate
    
    def __str__(self) -> str:
        out = self.getOutput()
        return 'mean:{:.4f}  covariance:{}  acceptance rate:{:.4f}'.format(out.mean(), np.cov(out.T), self.is_accepted.astype(np.float32).mean())
    



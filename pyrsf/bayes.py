
import gzip
import os
import sys
from time import time

import emcee
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib import gridspec
from scipy.stats import pearsonr
from six.moves import cPickle as pickle


class bayes_framework:
    """
    API for the Bayesian inference (inversion using a probabilistic random
    sampling approach)
    """

    msg_width = 30
    bayes_params = {}

    def __init__(self):
        self.bayes_params.update({"nsteps": 2000, "nthreads": 4, "nburn": 500})
        pass

    def log_prior(self):
        """
        Calculate a flat prior distribution. TODO: explore other priors
        Returns: log(1) = 0.0
        """

        return 0.0

    def log_likelihood(self, t, y):
        """
        Calculate the log likelihood of the data, given the model parameters
        Input: time, friction data
        Returns: log likelihood
        """

        params = self.params

        # TODO: calculate the data reciprocal weights around peak friction
        weights = 1.0
        sigma = weights * params["sigma"]

        if self.solver_mode == "step":
            model_result = self.forward(t, mode=self.solver_mode)
            mu_model = self.interpolate(t, model_result["t"], model_result["mu"])
        else:
            model_result = self.forward(t)
            mu_model = model_result["mu"]

        if np.isnan(mu_model[-1]):
            return -np.inf

        # Compute the likelihood of the data given the model parameters,
        # assuming that model-data mismatch is normally distributed with
        # standard deviation sigma
        logl = -0.5*np.sum(np.log(2*np.pi*sigma**2) + (y - mu_model)**2 / sigma**2)

        return logl

    def log_posterior(self, p, t, y):
        """
        Calculate the log posterior (\propto log prior + log likelihood),
        given the model parameters (a, b, Dc, and optionally k)
        Input: model parameters, time, friction data
        Returns: log posterior
        """

        self.unpack_params(p)
        params = self.params

        if params["a"] < 0 or params["Dc"] <= 0 or params["sigma"] <= 0 or params["k"] <= 0:
            return -np.inf

        return self.log_prior() + self.log_likelihood(t, y)

    def inv_bayes(self, p0, pickle_file="bayes_pickle.tar.gz"):
        """
        Perform the Bayesian inference (inversion), given the data. The initial
        guess is obtained from
        Input: initial guess, pickle output file
        Returns: summary statistocs of the sampled posterior distribution
        """

        self.unpack_params(p0)
        params = self.params
        bayes_params = self.bayes_params
        data = self.data

        ndim = len(p0)
        nwalkers = 4*ndim
        nsteps = bayes_params["nsteps"]
        nthreads = bayes_params["nthreads"]

        # Initiate random values between 0.5 and 1.5
        # TODO: initialise walkers with tight gaussian around p0
        starting_guess = 0.5 + np.random.random((nwalkers, ndim))

        # Multiply with inversion results
        for i, key in enumerate(self.inversion_params):
            starting_guess[:, i] *= params[key]

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_posterior,
            args=[data["t"], data["mu"]], threads=nthreads
        )

        print("Sampling posterior distribution with %d walkers..." % nwalkers)

        t0 = time()

        dN = int(nsteps//10)
        ETA_str = "--"

        try:
            for i, result in enumerate(sampler.sample(starting_guess, iterations=nsteps)):
                if i > 0 and i % dN == 0:
                    t_i = time()
                    inv_rate = (t_i-t0)/float(i)
                    todo = nsteps-i
                    ETA = todo*inv_rate
                    ETA_str = "%.2f s" % ETA

                n = int((self.msg_width + 1) * float(i) / nsteps)
                sys.stdout.write("\r[{0}{1}]\tETA: {2}".format('#' * n, ' ' * (self.msg_width - n), ETA_str))
        except pickle.PickleError:
            print("Python2.7 compatibility issue detected, switching from multithreaded to singlethreaded")
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, self.log_posterior,
                args=[data["t"], data["mu"]], threads=1
            )
            for i, result in enumerate(sampler.sample(starting_guess, iterations=nsteps)):
                if i > 0 and i % dN == 0:
                    t_i = time()
                    inv_rate = (t_i-t0)/float(i)
                    todo = nsteps-i
                    ETA = todo*inv_rate
                    ETA_str = "%.2f s" % ETA

                n = int((self.msg_width + 1) * float(i) / nsteps)
                sys.stdout.write("\r[{0}{1}]\tETA: {2}".format('#' * n, ' ' * (self.msg_width - n), ETA_str))

        sys.stdout.write("\n")

        t1 = time()
        print("MCMC execution time: %.2f" % (t1 - t0))

        self.chain = sampler.chain
        self.pickle_chain(pickle_file)
        stats = self.get_mcmc_stats().T

        return stats

    def pickle_chain(self, pickle_file):
        """
        Export the results of the Bayesian inference to disk using Python's
        pickle protocol
        :param pickle_file: the name of the output file
        :return:
        """

        output = {
            "params": self.params,
            "bayes_params": self.bayes_params,
            "data": self.data,
            "chain": self.chain,
        }

        print("Dumping pickles...")
        with gzip.GzipFile(pickle_file, "w") as f:
            pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)
        return True

    def unpickle_chain(self, pickle_file):
        print("Loading pickles...")
        if not os.path.isfile(pickle_file):
            print("Pickles not found!")
            return False
        try:
            with gzip.GzipFile(pickle_file, "r") as f:
                data = pickle.load(f)
        except ValueError as e:
            print("Exception '%s' caught, this is likely related to Python version incompatibility" % e)
            print("Re-run the Bayesian inference using the desired Python version. Will now exit...")
            exit()

        self.__dict__.update(data)
        return True

    def prune_chain(self):
        """
        Sometimes a few walkers get stuck in a local minimum. Prune those
        walkers astray from the sampling chain
        """

        chain = self.chain
        nburn = self.bayes_params["nburn"]

        stats = np.zeros((len(self.inversion_params), 2))

        for i, key in enumerate(self.inversion_params):
            param = chain[:, nburn:, i].reshape(-1)
            std = param.std()
            mean = param.mean()
            dx = np.abs(mean - param)
            param[dx > 2*std] = np.nan
            stats[i, 0] = np.nanmean(param)
            stats[i, 1] = np.nanstd(param)
            plt.plot(np.sort(dx)[::-1], ".")
            plt.axhline(2*std, ls="--", c="k")
            plt.show()

        return stats

    def get_mcmc_stats(self):
        """
        Calculate the mean and standard deviation of MCMC chain for each model
        parameter in the posterior distribution (after a certain burn-in)
        Returns: posterior distribution statistics
        """

        chain = self.chain

        nburn = self.bayes_params["nburn"]

        stats = np.zeros((len(self.inversion_params), 2))

        for i, key in enumerate(self.inversion_params):
            param = chain[:, nburn:, i].reshape(-1)
            stats[i, 0] = param.mean()
            stats[i, 1] = param.std()

        return stats

    def plot_mcmc_chain(self):
        """
        Plot the trace and distribution of each model parameter in the MCMC chain
        """

        chain = self.chain

        nburn = self.bayes_params["nburn"]
        ndim = chain.shape[2]
        nwalkers = chain.shape[0]

        gs = gridspec.GridSpec(ndim, 3)

        plt.figure(figsize=(10, 1.5*ndim))

        for i, key in enumerate(self.inversion_params):
            param = chain[:, nburn:, i].reshape(-1)

            plt.subplot(gs[i, :-1])
            for j in range(nwalkers):
                plt.plot(chain[j, :, i], lw=1.0, c="k", alpha=0.3)
            plt.axvline(nburn, ls="--", c="darkgray")
            plt.ylabel("%s" % key)
            if i == ndim-1:
                plt.xlabel("step")

            hist, bins = np.histogram(param, bins="auto")
            midbins = 0.5 * (bins[1:] + bins[:-1])

            plt.subplot(gs[i, -1])
            plt.plot(midbins, hist)
            plt.axvline(param.mean(), ls="--", c="k")

        plt.tight_layout()
        plt.show()

    def corner_plot(self):

        chain = self.chain
        ndim = chain.shape[2]-1
        nburn = self.bayes_params["nburn"]

        plt.figure(figsize=(10, 8))

        for i in range(ndim):
            param_i = chain[:, nburn:, i].reshape(-1)
            for j in range(i+1):
                ax = plt.subplot(ndim, ndim, 1+ndim*i+j)

                if i == j:
                    hist, bins = np.histogram(param_i, bins="auto")
                    midbins = 0.5*(bins[1:] + bins[:-1])
                    plt.plot(midbins, hist)
                    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.2e"))
                else:
                    param_j = chain[:, nburn:, j].reshape(-1)
                    r, p = pearsonr(param_i, param_j)
                    plt.plot(param_j, param_i, ".", ms=1, alpha=0.5)
                    plt.plot(np.median(param_j), np.median(param_i), "o", mew=1, mfc="r", mec="k")
                    plt.text(0.5, 0.9, "pearson r: %.2f" % r, transform=ax.transAxes, fontsize=9, ha="center")
                    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2e"))
                    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.2e"))

                if j == 0:
                    plt.ylabel(self.inversion_params[i])
                if i == ndim-1:
                    plt.xlabel(self.inversion_params[j])

        plt.tight_layout()
        plt.show()



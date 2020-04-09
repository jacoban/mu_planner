import math

# Z-values for various significance levels
ZTABLE = dict([(0.05, 1.645), (0.03, 1.88), (0.025, 1.96), (0.01, 2.33), (0.005, 2.58)])

def pr_gt(sample_func, prob=0.50, alpha=0.05, nsamples_init=30, nsamples_per_batch=10, nsamples_max=300):
    """
    Returns true if, according to a Z-test, the probability that the Bernoulli random variable is true is above threshold. The alternative hypothesis is that the probability is above threshold.
    """

    # `calcSlack` imposes a lower bound on the minimum of samples needed
    nsamples_min = math.ceil(math.log(1 / alpha) / (1 - prob))
    if nsamples_init < nsamples_min:
        nsamples_init = nsamples_min

    # H0: Pr(sample_func() = True) = prob
    # H1: Pr(sample_func() = True) > prob

    # nubmer of samples drawn so far
    k_total = 0
    # number of samples that test true
    k_true = 0
    # number of batches so far
    b = 0

    while True:
        for _ in range(nsamples_init if b == 0 else nsamples_per_batch):
            result = sample_func()
            if result:
                k_true += 1
            k_total += 1
        b += 1
        slack = calcSlack(alpha, k_total, k_true)
        if prob <= k_true / k_total - slack:
            return True # reject H0
        if k_total >= nsamples_max: # run out of samples
            return False # accept H0
        if prob > (nsamples_max - k_total + k_true) / nsamples_max - slack: # unable to reach the threshold even when all remaining samples test true
            return False # accept H0


def calcSlack(alpha, nsamples, nsamples_true):

    try:
        z = ZTABLE[alpha]
    except KeyError:
        raise ValueError("Significane level %f is not supported" % alpha)

    if nsamples_true == nsamples:
        # beware of an all-True situation, per the book
        return math.log(1 / alpha) / nsamples
    else:
        return z / nsamples * math.sqrt(nsamples_true - nsamples_true * nsamples_true / nsamples)

import math

def pr_gt(sample_func, prob=0.50, alpha=0.05, beta=0.05, slack_upper=0.01, slack_lower=0.01, nsamples_per_batch=10, nsamples_max=300, bounded_error_type=1):
    """
    Returns true if, according to Wald's Sequential Probability Ratio Test (SPRT), the probability that the Bernoulli random variable is true is above threshold. The alternative hypothesis is that the probability is above threshold.

    Args:
        sample_func: Probabilistic proposition to evaluate
        prob: Probability threshold to compare against.
        alpha: Upper bound on type 1 error rate (i.e., falsely return true).
        beta: Upper bound on type 2 error rate (i.e., falsely return false).
        slack_upper: Upper indifference region used by the SPRT
        slack_lower: Lower indifference region used by the SPRT
        nsamples_per_batch: Number of samples to draw per batch
        nsamples_max: Limit on the number of samples to draw (0 means no limit)
        bounded_error_type: Either 1 or 2. Means the type of error to be conservative about.
    """

    if prob >= 0.99:
        slack_lower = (1 - prob)*0.5
        slack_upper = slack_lower

    # H0: p = prob - slack
    prob0 = prob - slack_lower
    # H1: p = prob + slack
    prob1 = prob + slack_upper

    # to calculate the log likelihood ratio (LLR) of data seen so far
    LR0 = math.log(1 - prob1) - math.log(1 - prob0)
    LR1 = math.log(prob1) - math.log(prob0)
    calc_LLR = lambda k_total, k_true: \
        (k_total - k_true) * LR0 + k_true * LR1

    # reject H0 (and hence accept H1) if the log-likelihood >= B
    B = math.log((1 - beta) / alpha)
    # accept H0 if the log-likelihood <= A
    A = math.log(beta / (1 - alpha))

    assert (B > 0 and A < 0)

    # number of sample before first test is set to the smallest possible number of samples needed to pass/fail the test
    # Bigger alpha means smaller nsamples_min_pass
    # Bigger beta means smaller nsamples_min_fail
    nsamples_min_pass = math.ceil(B / LR1)
    nsamples_min_fail = math.ceil(A / LR0)
    nsamples_init = min(nsamples_min_pass, nsamples_min_fail)

    # number of samples drawn so far
    k_total = 0
    # number of samples that test true
    k_true = 0
    # number of batches so far
    b = 0

    while True:
        batch_size = nsamples_init if b == 0 else nsamples_per_batch
        for _ in range(batch_size):
            result = sample_func()
            k_total += 1
            if result:
                k_true += 1
        b += 1
        LLR = calc_LLR(k_total, k_true)

        # accept H0
        if LLR <= A:
            return False

        # reject H0
        if LLR >= B:
            # print("No collision after " + str((k_total, k_true)))
            return True

        if bounded_error_type == 1:
            # cannot decide even if all remaining samples test true
            if calc_LLR(nsamples_max, k_true + nsamples_max - k_total) < B:
                return False
        elif bounded_error_type == 2:
            # cannot decide even if all remaining samples test false
            if calc_LLR(nsamples_max, k_true) > A:
                return True
        else:
            raise ValueError("bounded_error_type must be either 1 or 2")

        # continue sampling if we can afford more samples
        if k_total < nsamples_max or nsamples_max == 0:
            continue
        else:
            if bounded_error_type == 1:
                return False
            elif bounded_error_type == 2:
                return True
            else:
                raise ValueError("bounded_error_type must be either 1 or 2")

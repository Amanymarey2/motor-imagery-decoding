#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
horseshoe_power_fit.py - descriptive fit of the horseshoe on power features,
all 30 subjects at once, to read the SIGN of the weights.

Target: 1 = right-hand imagery. Features: log band power, z-scored within subject.
Physiological prediction (write it down before running):
    right imagery suppresses mu/beta over the LEFT hemisphere (C3 side), so
    higher C3-side power should predict LEFT  -> negative weights on the left channels
    higher C4-side power should predict RIGHT -> positive weights on the right channels
Caveat: weights are not activation patterns. Neighbouring channels share signal, so
a single channel's weight can partly act as a contrast with its neighbours.
This is NOT a performance estimate (it uses all subjects).
"""
import jax.random as random
import numpy as np
import pandas as pd

from ard_cross_subject import names_per_set
from compare_features import feature_sets
from cross_subject import RIGHT, pooled
from horseshoe_cross_subject import P0, fit_posterior

if __name__ == "__main__":
    X, y, g = pooled(feature_sets())
    target = (y == RIGHT).astype(int)
    post, losses = fit_posterior(X["power_12ch"], target, P0["power_12ch"],
                                 random.PRNGKey(0))
    beta = np.asarray(post["beta"])
    out = pd.DataFrame({
        "feature": names_per_set()["power_12ch"],
        "mean": beta.mean(axis=0),
        "lo5": np.quantile(beta, 0.05, axis=0),
        "hi95": np.quantile(beta, 0.95, axis=0),
    })
    out["selected"] = (out.lo5 > 0) | (out.hi95 < 0)
    out.to_csv("data/processed/horseshoe_power_weights.csv", index=False)
    k = len(losses)
    print(f"ELBO loss at 2/3 and end: {losses[2 * k // 3]:.0f}, {losses[-1]:.0f}")
    print(out.sort_values("mean", key=lambda s: s.abs(), ascending=False)
             .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

# In[ ]:




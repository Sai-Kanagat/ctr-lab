# LinkedIn post drafts (not posted, Sai decides)

Numbers source: results/summary.json (test set, 1,998,311 impressions). Clicks are counted, not priced: no bids, costs or conversions are modelled. The top 1% figure in Variant B comes from results/summary.json, budget_curve, deepfm entry with k_pct 1 (13,559 clicks) against the random entry (3,044.78 expected clicks). The README does not state it.

Repo link is in place in both variants. Nothing has been posted.

## Variant A: the near tie

A plain logistic regression scored 0.751 AUC on 2 million ad impressions. The neural net scored 0.757. That gap is 0.006.

I'm a designer turned AI growth operator. Predicting ad clicks is the task Google Ads and every ad platform solve, so I tried it myself on the public Avazu mobile ad data from 2014 (not Google Ads data). Training used 8 million impressions with 22 anonymised fields, all on my MacBook Air.

Three models: logistic regression, LightGBM, DeepFM.
LightGBM came last at 0.737.

DeepFM also looks a little better where clicks are concentrated. Rank the test impressions by predicted click chance, take the top 10%, and DeepFM catches 26.7% of all clicks. Random picking catches 10%. That is a 2.67x lift. Logistic regression and LightGBM both land near 25% at the same cutoff, about 5,000 fewer clicks than DeepFM.

With a gap this small, I would test the simple model properly before choosing the heavier one.

Disclosure: one run, no confidence intervals. I picked DeepFM as the headline model by its test clicks at the 10% cutoff (it also has the best test AUC and log loss). Logistic regression tuning hit the edge of its grid. Click rate drifted from 17.5% in training to 15.2% in test. Details in the repo: https://github.com/Sai-Kanagat/ctr-lab

#adtech #machinelearning

## Variant B: the lift per impression

Buy the top 10% of ad impressions ranked by a model's predicted click rate and you capture 26.7% of all the clicks. Pick 10% of impressions at random and you capture 10% on average.

That is 2.67x the clicks of random picking, per impression, measured on 2 million held-out mobile ad impressions from the public 2014 Avazu dataset. It is the same task Google Ads and other ad platforms solve, but this is not Google Ads data and not their system.

I'm a designer turned AI growth operator. I trained three click predictors on my MacBook Air: logistic regression, LightGBM and DeepFM. DeepFM scored highest in my one run, with 0.757 AUC. Plain logistic regression landed at 0.751, a gap of 0.006, so I would not call it a proven win.

I plotted the budget curve as well as scoring AUC. The lift is biggest at small cutoffs, 4.45x at the top 1% for DeepFM (13,559 clicks against about 3,045 expected at random), and it shrinks steadily as the cutoff grows.

Disclosure: one run, no confidence intervals. I picked DeepFM as the headline model by its test clicks at the 10% cutoff (it also has the best test AUC and log loss). Logistic regression tuning hit the edge of its grid. Click rate drifted from 17.5% in training to 15.2% in test. Details in the repo: https://github.com/Sai-Kanagat/ctr-lab

#adtech #growth

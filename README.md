# Can simple rules beat the S&P 500? A pre-registered factor search

**Reed Ostrov, Denison University. Live at [reedsmoneymaker23.com/sp500](https://reedsmoneymaker23.com/sp500).**

## Question

Can any rule for tilting inside the S&P 500 beat the plain cap-weighted index after costs?

## Design

- 23 testable angles, each registered in `trials.csv` before testing. 84 hypotheses ran in the top-4% tournament batches.
- Training data: 1996 to 2015. The gate required t ≥ 3, deflated Sharpe ≥ 0.95 at the full trial count, the right sign in ≥ 60% of 5-year windows, and Fama-French 5-factor plus momentum alpha t ≥ 2.
- Holdout: January 2016 to August 2026, 128 months, opened once, net of costs.

## Result

**0 of 23 angles passed the gate.** The final portfolio is the plain index plus one implementation rule.

| Portfolio, 2016 to 2026 holdout | Annualized return |
|---|---|
| This system | 15.22% |
| Reconstructed index | 15.20% |
| S&P 500 total return | 15.16% |
| VOO | 15.13% |

The 0.08 point gap over VOO is reconstruction noise (0.39% tracking error). 8 of 13 angles with positive pre-holdout performance lost more than half of it out of sample, the usual sign of overfitting.

## SEC contact email

The SEC requires a contact email in the User-Agent of every EDGAR request. Downloads stop with an error until you set yours (there is no default):

    export SEC_CONTACT_EMAIL="you@example.com"

The cron script reads it from a local `.env` file (`SEC_CONTACT_EMAIL=you@example.com`), which git ignores.

Full report: [reports/](reports/). License: MIT. Cite via [CITATION.cff](CITATION.cff).

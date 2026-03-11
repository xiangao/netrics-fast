# netrics-fast

Memory-efficient dyadic regression with bias-corrected dyadic-robust (DR_bc) standard errors, following Graham (forthcoming, *Handbook of Econometrics*).

Fast reimplementation of `netrics.dyadic_regression` using chunked O(nK) scatter-add Hajek projection — handles 100M+ dyads without materializing the full score matrix.

## Install

```bash
pip install -e .
```

## Usage

```python
from netrics_fast import dyadic_regression, print_coef

result = dyadic_regression(Y, R, id_i, id_j, directed=False, cov="DR_bc")
print_coef(result["beta"], result["vcov"], result["var_names"])
```

## Reference

Graham, B. S. (forthcoming). "Network Data." *Handbook of Econometrics*, Volume 7A.

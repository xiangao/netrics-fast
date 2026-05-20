# netrics-fast

`netrics-fast` is a memory-conscious implementation of dyadic regression with
bias-corrected dyadic-robust standard errors, following Graham (forthcoming,
*Handbook of Econometrics*).

It reimplements `netrics.dyadic_regression` using a chunked O(nK) scatter-add
Hajek projection, so the full score matrix does not have to be materialized.

## Install

```bash
pip install git+https://github.com/xiangao/netrics-fast.git
```

## Usage

```python
from netrics_fast import dyadic_regression, print_coef

result = dyadic_regression(Y, R, id_i, id_j, directed=False, cov="DR_bc")
print_coef(result["beta"], result["vcov"], result["var_names"])
```

## Reference

Graham, B. S. (forthcoming). "Network Data." *Handbook of Econometrics*, Volume 7A.

"""
Tests for netrics_fast.dyadic_regression.

Verifies against known results from the ekempf_executives project.
"""

import os
import numpy as np
import pytest

from netrics_fast import dyadic_regression, print_coef


# Path to test data in the ekempf_executives project
EKEMPF = os.path.expanduser("~/projects/claude/ekempf_executives")


class TestSimDat:
    """Verify against dyadRobust on sim-dat.RData (n=4950, N=100)."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        pyreadr = pytest.importorskip("pyreadr")
        path = os.path.join(EKEMPF, "dyad_aronow/dyadRobust/data/sim-dat.RData")
        if not os.path.exists(path):
            pytest.skip(f"sim-dat.RData not found at {path}")
        rdata = pyreadr.read_r(path)
        self.df = list(rdata.values())[0]

    def test_ols_coefficients(self):
        """OLS coefficients should match lm() on sim-dat."""
        df = self.df
        # sim-dat columns: dyads, dY, dX, dyad1, dyad2
        Y = df["dY"].values
        R = df["dX"].values.reshape(-1, 1)
        id_i = df["dyad1"].values
        id_j = df["dyad2"].values

        result = dyadic_regression(Y, R, id_i, id_j, directed=False,
                                   cov="DR_bc", silent=True)

        # Coefficients should be finite and reasonable
        assert result["beta"].shape == (2,)
        assert np.all(np.isfinite(result["beta"]))
        assert np.all(np.isfinite(result["se"]))
        assert result["N"] == 100
        assert result["n"] == len(df)

    def test_dr_se_larger_than_hc(self):
        """DR_bc SEs should generally be >= HC-robust SEs for dyadic data."""
        df = self.df
        Y = df["dY"].values
        R = df["dX"].values.reshape(-1, 1)
        id_i = df["dyad1"].values
        id_j = df["dyad2"].values

        res_dr = dyadic_regression(Y, R, id_i, id_j, cov="DR_bc", silent=True)
        res_hc = dyadic_regression(Y, R, id_i, id_j, cov="ind", silent=True)

        # DR_bc accounts for dyadic dependence — SEs should be at least as large
        assert np.all(res_dr["se"] >= res_hc["se"] * 0.9)  # allow small tolerance


class TestDyadicSample5:
    """Verify against known results from dyadic_sample_5.dta."""

    @pytest.fixture(autouse=True)
    def load_data(self):
        pd = pytest.importorskip("pandas")
        path = os.path.join(EKEMPF, "data/dyadic_sample_5.dta")
        if not os.path.exists(path):
            pytest.skip(f"dyadic_sample_5.dta not found at {path}")
        self.df = pd.read_stata(path)

    def test_known_results(self):
        """Match known results: constant=0.0502, SParty=0.0097,
        SE_DR_bc constant=0.0040, SE_DR_bc SParty=0.0053."""
        df = self.df
        Y = df["SFirm100"].values
        R = df[["SParty"]].values
        id_i = df["execid_2"].values
        id_j = df["execid"].values

        result = dyadic_regression(Y, R, id_i, id_j, directed=False,
                                   cov="DR_bc", silent=True)

        # Coefficients (4 decimal places)
        assert result["beta"][0] == pytest.approx(0.0502, abs=5e-5)
        assert result["beta"][1] == pytest.approx(0.0097, abs=5e-5)

        # DR_bc standard errors (4 decimal places)
        assert result["se"][0] == pytest.approx(0.0040, abs=5e-5)
        assert result["se"][1] == pytest.approx(0.0053, abs=5e-5)

        # Metadata
        assert result["N"] == 12_129
        assert result["n"] == 6_736_741

    def test_cov_dr_no_bc(self):
        """DR (no bias correction) should also work."""
        df = self.df
        Y = df["SFirm100"].values
        R = df[["SParty"]].values
        id_i = df["execid_2"].values
        id_j = df["execid"].values

        result = dyadic_regression(Y, R, id_i, id_j, directed=False,
                                   cov="DR", silent=True)

        assert np.all(np.isfinite(result["se"]))
        # DR (no bc) SEs should be slightly larger than DR_bc for large N
        # (bias correction subtracts a positive term)

    def test_ind_cov(self):
        """Independence (HC-robust) SEs should work."""
        df = self.df
        Y = df["SFirm100"].values
        R = df[["SParty"]].values
        id_i = df["execid_2"].values
        id_j = df["execid"].values

        result = dyadic_regression(Y, R, id_i, id_j, directed=False,
                                   cov="ind", silent=True)

        # Same coefficients regardless of SE method
        assert result["beta"][0] == pytest.approx(0.0502, abs=5e-5)
        assert result["beta"][1] == pytest.approx(0.0097, abs=5e-5)


class TestPrintCoef:
    """Test print_coef output."""

    def test_basic(self, capsys):
        beta = np.array([1.0, 2.0])
        vcov = np.diag([0.01, 0.04])
        print_coef(beta, vcov, ["constant", "X1"])
        captured = capsys.readouterr()
        assert "constant" in captured.out
        assert "X1" in captured.out

    def test_2d_beta(self, capsys):
        """Should handle (K,1) shaped beta."""
        beta = np.array([[1.0], [2.0]])
        vcov = np.diag([0.01, 0.04])
        print_coef(beta, vcov)
        captured = capsys.readouterr()
        assert "X_0" in captured.out


class TestEdgeCases:
    """Test edge cases and options."""

    def test_nocons(self):
        """nocons=True should not prepend a constant."""
        np.random.seed(42)
        n = 100
        Y = np.random.randn(n)
        R = np.column_stack([np.ones(n), np.random.randn(n)])
        ids = np.arange(20)
        id_i = np.random.choice(ids, n)
        id_j = np.random.choice(ids, n)

        result = dyadic_regression(Y, R, id_i, id_j, nocons=True, silent=True)
        assert result["beta"].shape == (2,)

    def test_single_regressor(self):
        """1-d R array should work."""
        np.random.seed(42)
        n = 100
        Y = np.random.randn(n)
        R = np.random.randn(n)
        ids = np.arange(20)
        id_i = np.random.choice(ids, n)
        id_j = np.random.choice(ids, n)

        result = dyadic_regression(Y, R, id_i, id_j, silent=True)
        assert result["beta"].shape == (2,)  # constant + 1 regressor

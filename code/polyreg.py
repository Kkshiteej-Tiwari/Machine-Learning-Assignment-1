"""Core utilities: polynomial feature construction and ridge-regularised
polynomial regression with efficient cross-validation over a whole
regularisation path (one SVD per fold, every alpha evaluated for free)."""
import itertools
import numpy as np
from numpy.polynomial import legendre


# ---------------------------------------------------------------- features
def exponents(n_features, degree):
    """All multi-indices (a_1..a_n) with sum(a) <= degree, constant term first."""
    out = []
    for d in range(degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_features), d):
            e = [0] * n_features
            for j in combo:
                e[j] += 1
            out.append(tuple(e))
    return np.array(out, dtype=int)


def _legendre_table(x, degree):
    """P_0..P_degree evaluated at x, shape (len(x), degree+1); scaled so that
    each P_k has unit mean square on [-1, 1]."""
    tab = legendre.legvander(x, degree)
    return tab * np.sqrt(2 * np.arange(degree + 1) + 1)


def poly_features(X, degree, basis="legendre"):
    """Design matrix of every polynomial term of total degree <= `degree`.

    basis="monomial": prod_j x_j^a_j (the textbook form).
    basis="legendre": prod_j P_{a_j}(x_j). This spans exactly the same space of
    polynomials (a change of basis), but is far better conditioned on [-1, 1],
    which matters for degree 10-20.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    E = exponents(p, degree)
    if basis == "monomial":
        tabs = [np.vander(X[:, j], degree + 1, increasing=True) for j in range(p)]
    else:
        tabs = [_legendre_table(X[:, j], degree) for j in range(p)]
    Phi = np.ones((n, len(E)))
    for j in range(p):
        Phi *= tabs[j][:, E[:, j]]
    return Phi


# ---------------------------------------------------------------- ridge
class RidgePath:
    """Ridge regression (unpenalised intercept) solved via SVD so that any
    number of alphas can be evaluated from a single decomposition."""

    def __init__(self, Phi, y):
        # column 0 is the constant term -> handled by centring instead
        A = Phi[:, 1:]
        self.mu = A.mean(0)
        self.ybar = y.mean()
        U, s, Vt = np.linalg.svd(A - self.mu, full_matrices=False)
        self.U, self.s, self.Vt = U, s, Vt
        self.Uty = U.T @ (y - self.ybar)

    def coef(self, alpha):
        d = self.s / (self.s ** 2 + alpha)
        w = self.Vt.T @ (d * self.Uty)
        b = self.ybar - self.mu @ w
        return np.concatenate([[b], w])

    def predict_many(self, Phi_new, alphas):
        """Predictions for every alpha at once: shape (n_new, n_alphas)."""
        A = Phi_new[:, 1:] - self.mu
        AV = A @ self.Vt.T                       # (n_new, r)
        D = self.s[:, None] / (self.s[:, None] ** 2 + np.asarray(alphas)[None, :])
        return self.ybar + AV @ (D * self.Uty[:, None])


def cv_ridge(X, y, degree, alphas, folds, weights=None, basis="legendre"):
    """K-fold CV of ridge polynomial regression for one degree.

    Returns (n_folds, n_alphas) array of (optionally weighted) validation MSE.
    `weights` re-weights validation points (importance weighting for the
    train->test distribution shift); training itself is unweighted.
    """
    Phi = poly_features(X, degree, basis)
    out = np.empty((len(folds), len(alphas)))
    for k, (tr, va) in enumerate(folds):
        model = RidgePath(Phi[tr], y[tr])
        P = model.predict_many(Phi[va], alphas)
        err = (P - y[va, None]) ** 2
        w = np.ones(len(va)) if weights is None else weights[va]
        out[k] = (w[:, None] * err).sum(0) / w.sum()
    return out


def kfold(n, k, seed):
    idx = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(idx, k)
    return [(np.concatenate(parts[:i] + parts[i + 1:]), parts[i]) for i in range(k)]


def boundary_count(X):
    """Number of coordinates of each row that sit exactly on the +-1 boundary."""
    return (np.abs(np.asarray(X)) >= 1 - 1e-12).sum(1)


def shift_weights(X_train, X_test):
    """Importance weights w(k) = p_test(k) / p_train(k), where k is the number
    of clipped (+-1) coordinates in a row. Makes CV error mimic the test mix."""
    kt, ks = boundary_count(X_train), boundary_count(X_test)
    m = X_train.shape[1] + 1
    p_tr = np.bincount(kt, minlength=m) / len(kt)
    p_te = np.bincount(ks, minlength=m) / len(ks)
    ratio = np.where(p_tr > 0, p_te / np.maximum(p_tr, 1e-12), 0.0)
    return ratio[kt]


# ---------------------------------------------------------------- lasso
def degree_scale(n_features, degree, gamma):
    """Column scale (1 + total degree of term)^(-gamma/2) for every non-constant
    term. Scaling a column by s is equivalent to multiplying that coefficient's
    penalty by 1/s, so gamma > 0 penalises high-order terms more strongly
    (a smoothness prior). gamma = 0 recovers the ordinary penalty."""
    E = exponents(n_features, degree)[1:]
    return (1.0 + E.sum(1)) ** (-gamma / 2.0)


def cv_lasso(X, y, degree, gamma, lambdas, folds, weights=None, basis="legendre"):
    """K-fold CV of Lasso polynomial regression along a lambda path.
    Returns (n_folds, n_lambdas) array of (optionally weighted) validation MSE."""
    from sklearn.linear_model import lasso_path
    Phi = poly_features(X, degree, basis)[:, 1:] * degree_scale(X.shape[1], degree, gamma)
    out = np.empty((len(folds), len(lambdas)))
    for k, (tr, va) in enumerate(folds):
        mu, ym = Phi[tr].mean(0), y[tr].mean()
        A = Phi[tr] - mu
        # precomputed Gram matrix: ~100x faster coordinate descent when
        # the number of terms is comparable to the number of rows
        _, C, _ = lasso_path(A, y[tr] - ym, alphas=lambdas, precompute=A.T @ A,
                             max_iter=5000, tol=1e-4)
        P = ym + (Phi[va] - mu) @ C
        err = (P - y[va, None]) ** 2
        w = np.ones(len(va)) if weights is None else weights[va]
        out[k] = (w[:, None] * err).sum(0) / w.sum()
    return out


# ---------------------------------------------------------------- final model
class PolyModel:
    """A single polynomial regression model y = sum_t c_t * phi_t(x) over all
    terms phi_t of total degree <= `degree`.

    method="ridge": L2 penalty `lam`.   method="lasso": L1 penalty `lam`.
    `gamma` makes the penalty grow with the order of the term (see degree_scale).
    """

    def __init__(self, degree, method="ridge", lam=1.0, gamma=0.0, basis="legendre"):
        self.degree, self.method, self.lam = degree, method, lam
        self.gamma, self.basis = gamma, basis

    def _design(self, X):
        Phi = poly_features(X, self.degree, self.basis)
        Phi[:, 1:] *= degree_scale(X.shape[1], self.degree, self.gamma)
        return Phi

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        Phi = self._design(X)
        if self.method == "ridge":
            self.coef_ = RidgePath(Phi, y).coef(self.lam)
        else:
            from sklearn.linear_model import Lasso
            A = Phi[:, 1:]
            mu, ym = A.mean(0), y.mean()
            A = A - mu
            m = Lasso(alpha=self.lam, fit_intercept=False, precompute=A.T @ A,
                      max_iter=100000, tol=1e-6)
            m.fit(A, y - ym)
            self.coef_ = np.concatenate([[ym - mu @ m.coef_], m.coef_])
        self.n_features_ = X.shape[1]
        return self

    def predict(self, X):
        return self._design(np.asarray(X, float)) @ self.coef_

    def n_nonzero(self):
        return int((np.abs(self.coef_) > 1e-10).sum())

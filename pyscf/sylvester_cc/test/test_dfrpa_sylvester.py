from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from pyscf import gto, scf, sylvester_cc
from pyscf.sylvester_cc.dfrpa_sylvester import DFRPASylvester
from pyscf.sylvester_cc.laplace_quadrature import LaplaceQuadrature


def _bare_solver(eris=None, chebyshev_tol=1.0e-15):
    solver = DFRPASylvester.__new__(DFRPASylvester)
    solver._eris = eris
    solver.ll_chebyshev_tol = chebyshev_tol
    solver.ll_chebyshev_max_degree = 500
    solver.ll_xvo_t2_mode = "dense"
    solver.ll_exchange_screen_tol = 0.0
    solver.ll_exchange_screen_mode = "bound"
    solver.ll_exchange_q_block_size = 2
    solver.ll_exchange_rank_block_size = 3
    solver.ll_exchange_check_interval = 1
    solver.ll_exchange_screen_refresh = 0
    solver._exchange_screen_history = []
    solver._exchange_screen_selection = None
    solver._exchange_screen_calls = 0
    solver.verbose = 0
    return solver


def _random_factors(seed=12):
    rng = np.random.default_rng(seed)
    naux, nlap, nvir, nocc = 3, 4, 5, 2
    Y = 0.1 * rng.standard_normal((naux, nlap, nvir, nocc))
    return rng, Y


def test_hybrid_Xvo_t2_matches_dense_contraction():
    rng, Y = _random_factors()
    Lov = rng.standard_normal((Y.shape[0], Y.shape[-1], Y.shape[-2]))
    solver = _bare_solver(SimpleNamespace(Lov=Lov))

    t2 = solver.reconstruct_t2(Y)
    t2_antisym = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    reference = np.einsum("Lkc,ikac->Lai", Lov, t2_antisym)

    assert np.allclose(solver.get_Xvo_t2(Y), reference, atol=1.0e-13)


def test_optional_factorized_Xvo_t2_matches_dense_contraction():
    rng, Y = _random_factors(seed=13)
    Lov = rng.standard_normal((Y.shape[0], Y.shape[-1], Y.shape[-2]))
    solver = _bare_solver(SimpleNamespace(Lov=Lov))
    solver.ll_xvo_t2_mode = "factorized"

    t2 = solver.reconstruct_t2(Y)
    reference = np.einsum(
        "Lkc,ikac->Lai", Lov, 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    )

    assert np.allclose(solver.get_Xvo_t2(Y), reference, atol=2.0e-13)


def test_blocked_exchange_matches_unblocked_dense_contraction():
    rng, Y = _random_factors(seed=29)
    Lov = rng.standard_normal((Y.shape[0], Y.shape[-1], Y.shape[-2]))
    solver = _bare_solver(SimpleNamespace(Lov=Lov))
    Y_flat = solver._flatten_factors(Y)

    reference = np.einsum(
        "Lkc,Rci,Rak->Lai", Lov, Y_flat, Y_flat, optimize=True
    )
    result = solver.get_Xvo_t2_exchange(Y)

    assert np.allclose(result, reference, atol=2.0e-13)
    assert solver._exchange_screen_stats["evaluated_blocks"] == (
        solver._exchange_screen_stats["total_blocks"]
    )
    assert solver._exchange_screen_stats["skipped_blocks"] == 0


def test_zero_factors_avoid_exchange_blocks():
    rng, Y = _random_factors(seed=31)
    Lov = rng.standard_normal((Y.shape[0], Y.shape[-1], Y.shape[-2]))
    solver = _bare_solver(
        SimpleNamespace(Lov=Lov, Lvo=Lov.transpose(0, 2, 1))
    )

    result = solver.get_Xvo_t2(np.zeros_like(Y))

    assert np.count_nonzero(result) == 0
    assert solver._exchange_screen_stats["zero_factors"]
    assert solver._exchange_screen_stats["evaluated_blocks"] == 0


def test_dominant_first_exchange_screening_has_controlled_error():
    rng = np.random.default_rng(37)
    naux, nrank, nvir, nocc = 5, 9, 4, 2
    base_L = np.abs(rng.standard_normal((nocc, nvir)))
    base_Y = np.abs(rng.standard_normal((nvir, nocc)))
    L_scale = 10.0 ** -np.arange(naux)
    Y_scale = 10.0 ** (-0.5 * np.arange(nrank))
    Lov = L_scale[:, None, None] * base_L[None]
    Y = Y_scale[:, None, None] * base_Y[None]

    exact = np.einsum("Lkc,Rci,Rak->Lai", Lov, Y, Y, optimize=True)
    solver = _bare_solver(SimpleNamespace(Lov=Lov))
    solver.ll_exchange_screen_tol = 1.0e-3
    solver.ll_exchange_q_block_size = 1
    solver.ll_exchange_rank_block_size = 1
    screened = solver.get_Xvo_t2_exchange(Y)
    stats = solver._exchange_screen_stats

    relative_error = np.linalg.norm(screened - exact) / np.linalg.norm(exact)
    assert stats["converged"]
    assert stats["evaluated_blocks"] < stats["total_blocks"]
    assert stats["relative_error_bound"] <= solver.ll_exchange_screen_tol
    assert relative_error <= stats["relative_error_bound"] + 1.0e-14
    assert stats["largest_block_bound"] >= stats["smallest_evaluated_bound"]


def test_calibrated_exchange_screening_reuses_dominant_list():
    rng = np.random.default_rng(41)
    naux, nrank, nvir, nocc = 5, 9, 4, 2
    base_L = np.abs(rng.standard_normal((nocc, nvir)))
    base_Y = np.abs(rng.standard_normal((nvir, nocc)))
    Lov = 10.0 ** -np.arange(naux)[:, None, None] * base_L[None]
    Y = 10.0 ** (-0.5 * np.arange(nrank))[:, None, None] * base_Y[None]
    exact = np.einsum("Lkc,Rci,Rak->Lai", Lov, Y, Y, optimize=True)

    solver = _bare_solver(SimpleNamespace(Lov=Lov))
    solver.ll_exchange_screen_mode = "calibrated"
    solver.ll_exchange_screen_tol = 1.0e-3
    solver.ll_exchange_q_block_size = 1
    solver.ll_exchange_rank_block_size = 1
    first = solver.get_Xvo_t2_exchange(Y)
    calibration = solver._exchange_screen_stats
    second = solver.get_Xvo_t2_exchange(Y)
    reuse = solver._exchange_screen_stats

    first_error = np.linalg.norm(first - exact) / np.linalg.norm(exact)
    assert calibration["calibration_build"]
    assert calibration["calibration_relative_error"] <= 1.0e-3
    assert first_error <= 1.0e-3
    assert calibration["retained_blocks"] < calibration["total_blocks"]
    assert not reuse["calibration_build"]
    assert reuse["evaluated_blocks"] == calibration["retained_blocks"]
    assert np.allclose(second, first, atol=1.0e-14)


def test_factorized_omega0_matches_dense_source():
    rng, Y = _random_factors(seed=14)
    naux, _, nvir, nocc = Y.shape
    eris = SimpleNamespace(
        Lov=rng.standard_normal((naux, nocc, nvir)),
        Lvo=rng.standard_normal((naux, nvir, nocc)),
        Loo=rng.standard_normal((naux, nocc, nocc)),
        Lvv=rng.standard_normal((naux, nvir, nvir)),
        fov=rng.standard_normal((nocc, nvir)),
    )
    solver = _bare_solver(eris)
    t1 = 0.1 * rng.standard_normal((nocc, nvir))
    Fov = rng.standard_normal((nocc, nvir))
    X, _, Xvo, Xvo_t2 = solver.get_X_factorized(t1, Y)

    t2 = solver.reconstruct_t2(Y)
    t2_antisym = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    reference = eris.fov.T.copy()
    reference -= np.einsum("Laj,Lji->ai", Xvo, eris.Loo)
    reference -= np.einsum("Laj,Lji->ai", Xvo_t2, eris.Loo)
    reference += np.einsum("Lai,L->ai", eris.Lvo, X)
    reference += np.einsum("Lae,Lei->ai", eris.Lvv, Xvo_t2)
    reference += np.einsum("ijab,jb->ai", t2_antisym, Fov)

    result = solver.get_omega0_factorized(X, Xvo, Xvo_t2, Fov, Y)
    assert np.allclose(result, reference, atol=2.0e-12)


def test_singles_laplace_uses_actual_nonsymmetric_actions():
    rng = np.random.default_rng(8)
    Foo = rng.standard_normal((2, 2)) - 4.0 * np.eye(2)
    Fvv = rng.standard_normal((4, 4)) + 2.0 * np.eye(4)
    omega0 = rng.standard_normal((4, 2))
    quad = LaplaceQuadrature(
        exponents=np.array([0.03, 0.2, 0.9]),
        weights=np.array([0.2, 0.4, 0.1]),
        ymin=0.1,
        ymax=20.0,
    )
    solver = _bare_solver()

    reference = -sum(
        weight * expm(-alpha * Fvv) @ omega0 @ expm(alpha * Foo)
        for alpha, weight in zip(quad.exponents, quad.weights)
    )
    result = solver.solve_singles_laplace(omega0, Foo, Fvv, quad)

    assert not np.allclose(Foo, Foo.T)
    assert not np.allclose(Fvv, Fvv.T)
    assert np.allclose(result, reference, atol=2.0e-12)


def test_doubles_factors_match_explicit_matrix_exponentials():
    rng = np.random.default_rng(21)
    Foo = rng.standard_normal((2, 2)) - 3.0 * np.eye(2)
    Fvv = rng.standard_normal((4, 4)) + 2.0 * np.eye(4)
    Jvo = rng.standard_normal((3, 4, 2))
    quad = LaplaceQuadrature(
        exponents=np.array([0.04, 0.3, 0.7]),
        weights=np.array([0.3, 0.2, 0.05]),
        ymin=0.1,
        ymax=20.0,
    )
    solver = _bare_solver()

    reference = np.stack(
        [
            np.sqrt(weight)
            * np.einsum(
                "ab,Lbi,ij->Laj",
                expm(-alpha * Fvv),
                Jvo,
                expm(alpha * Foo),
            )
            for alpha, weight in zip(quad.exponents, quad.weights)
        ],
        axis=1,
    )
    result = solver.get_sylvester_laplace_matrix_factors(
        Jvo, Foo, Fvv, quad
    )

    assert np.allclose(result, reference, atol=2.0e-12)


def test_reconstructed_t2_has_pair_symmetry():
    _, Y = _random_factors(seed=17)
    t2 = DFRPASylvester.reconstruct_t2(Y)
    assert np.allclose(t2, t2.transpose(1, 0, 3, 2), atol=1.0e-14)


def test_nonpositive_dressed_gap_fails_clearly():
    solver = _bare_solver()
    quad = LaplaceQuadrature(
        exponents=np.array([0.1]),
        weights=np.array([1.0]),
        ymin=0.01,
        ymax=10.0,
    )
    with pytest.raises(ValueError, match="dressed orbital gaps must be positive"):
        solver._validate_dressed_gaps(
            np.array([[1.0]]), np.array([[0.0]]), quad, 1, "singles"
        )


def test_dressed_gap_outside_quadrature_fails_clearly():
    solver = _bare_solver()
    quad = LaplaceQuadrature(
        exponents=np.array([0.1]),
        weights=np.array([1.0]),
        ymin=1.0,
        ymax=2.0,
    )
    with pytest.raises(ValueError, match="does not cover required gaps"):
        solver._validate_dressed_gaps(
            np.array([[-1.0]]), np.array([[2.0]]), quad, 1, "singles"
        )


def test_canonical_water_energy_matches_dense_dfrpa():
    laplace_root = (
        Path(__file__).resolve().parents[3] / "external" / "laplace-minimax"
    )
    if not (laplace_root / "data" / "init_para.txt").is_file():
        pytest.skip("optional laplace-minimax tables are unavailable")

    mol = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        verbose=0,
    )
    mf = scf.RHF(mol).density_fit().run(conv_tol=1.0e-12)
    convergence = 1.0e-8

    reference = sylvester_cc.DFRPA(
        mf, ll_con_tol=convergence, ll_max_its=100
    )
    t1_reference, t2_reference = reference.kernel()
    reference_energy = reference.energy(t1_reference, t2_reference)

    factorized = sylvester_cc.DFRPASylvester(
        mf,
        ll_laplace_root=laplace_root,
        ll_laplace_npoints=16,
        ll_con_tol=convergence,
        ll_max_its=100,
        ll_chebyshev_tol=1.0e-14,
    )
    factorized.kernel()

    assert factorized.converged
    assert abs(factorized.e_corr - reference_energy) < 1.0e-7

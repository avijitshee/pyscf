import unittest
from unittest import mock

import numpy as np

from pyscf import sylvester_cc
from pyscf.sylvester_cc import df_eri, dfrccsd, dfrpa, dfrpa_sylvester
from pyscf.sylvester_cc import dfrpa_triples
from pyscf.sylvester_cc.dfrpa_triples import DFRPA


class LadderContractionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.t2 = rng.standard_normal((2, 2, 3, 3))
        self.Joo = rng.standard_normal((5, 2, 2))
        self.Jvv = rng.standard_normal((5, 3, 3))

    @staticmethod
    def make_solver(**kwargs):
        mf = mock.Mock()
        mf.with_df = object()
        mf.mol.nelec = (2, 2)
        mf.mol.nao = 5
        eris = mock.Mock()
        eris.Loo = np.empty((5, 2, 2))
        eris.Lvv = np.empty((5, 3, 3))
        return DFRPA(mf, eris, **kwargs)

    def test_active_jvv_is_the_default_mode(self):
        solver = self.make_solver(act_particle=np.array([0, 2]))
        self.assertEqual(solver.ladder_mode, "active_jvv")

    def test_only_active_and_full_modes_are_accepted(self):
        with self.assertRaisesRegex(ValueError, "active_jvv.*full"):
            self.make_solver(
                act_particle=np.array([0, 2]), ladder_mode="mixed_ao"
            )

        solver = self.make_solver(ladder_mode="full")
        self.assertEqual(solver.ladder_mode, "full")

    def test_active_mode_requires_particle_indices(self):
        with self.assertRaisesRegex(ValueError, "requires act_particle"):
            self.make_solver()

    def test_full_contraction_matches_explicit_einsum(self):
        reference = np.einsum(
            "Lmj,Lbe,imae->ijab", self.Joo, self.Jvv, self.t2
        )
        result = DFRPA.ladder_contraction_full(
            self.t2, self.Joo, self.Jvv
        )
        np.testing.assert_allclose(result, reference, atol=1.0e-13)

    def test_active_jvv_matches_full_contraction_with_masked_jvv(self):
        active = np.array([0, 2])
        Jvv_masked = np.zeros_like(self.Jvv)
        Jvv_masked[:, active[:, None], active] = self.Jvv[
            :, active[:, None], active
        ]
        reference = DFRPA.ladder_contraction_full(
            self.t2, self.Joo, Jvv_masked
        )

        result_from_full = DFRPA.ladder_contraction_active_jvv(
            self.t2, self.Joo, self.Jvv, active
        )
        result_from_active = DFRPA.ladder_contraction_active_jvv(
            self.t2,
            self.Joo,
            self.Jvv[:, active][:, :, active],
            active,
        )

        np.testing.assert_allclose(result_from_full, reference, atol=1.0e-13)
        np.testing.assert_allclose(
            result_from_active, reference, atol=1.0e-13
        )
        np.testing.assert_allclose(result_from_full[:, :, :, 1], 0.0)

    def test_active_index_validation(self):
        result = DFRPA._validate_active_indices(
            np.array([True, False, True]), 3, "act_particle"
        )
        np.testing.assert_array_equal(result, (0, 2))

        with self.assertRaisesRegex(ValueError, "duplicate"):
            DFRPA._validate_active_indices(
                np.array([1, 1]), 3, "act_particle"
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            DFRPA._validate_active_indices(
                np.array([3]), 3, "act_particle"
            )

    def test_get_J_builds_only_the_active_virtual_block(self):
        rng = np.random.default_rng(47)
        solver = self.make_solver(act_particle=np.array([0, 2]))
        solver._eris.Loo = rng.standard_normal((5, 2, 2))
        solver._eris.Lvo = rng.standard_normal((5, 3, 2))
        solver._eris.Lov = rng.standard_normal((5, 2, 3))
        solver._eris.Lvv = rng.standard_normal((5, 3, 3))
        t1 = rng.standard_normal((2, 3))
        Xoo = rng.standard_normal((5, 2, 2))
        Xvo = rng.standard_normal((5, 3, 2))
        Xvo_t2 = rng.standard_normal((5, 3, 2))

        _, _, Jvv = solver.get_J(Xoo, Xvo, Xvo_t2, t1)
        Jvv_full = solver._eris.Lvv - np.einsum(
            "Lkb,ka->Lab", solver._eris.Lov, t1
        )
        active = solver.act_particle

        self.assertEqual(Jvv.shape, (5, 2, 2))
        np.testing.assert_allclose(
            Jvv, Jvv_full[:, active][:, :, active], atol=1.0e-13
        )

    def test_active_dispatch_records_dimensions(self):
        solver = self.make_solver(
            act_hole=np.array([0]), act_particle=np.array([0, 2])
        )
        result = solver.ladder_contraction(self.t2, self.Joo, self.Jvv)
        reference = solver.ladder_contraction_active_jvv(
            self.t2, self.Joo, self.Jvv, solver.act_particle
        )

        np.testing.assert_allclose(result, reference, atol=1.0e-13)
        self.assertEqual(solver.active_ladder_report["nact_hole"], 1)
        self.assertEqual(solver.active_ladder_report["nact_particle"], 2)

    def test_full_dispatch_uses_reference_contraction(self):
        solver = self.make_solver(ladder_mode="full")
        result = solver.ladder_contraction(self.t2, self.Joo, self.Jvv)
        reference = solver.ladder_contraction_full(
            self.t2, self.Joo, self.Jvv
        )
        np.testing.assert_allclose(result, reference, atol=1.0e-13)

    def test_all_virtuals_active_reproduces_full_amplitude_update(self):
        rng = np.random.default_rng(53)
        mf = mock.Mock()
        mf.with_df = object()
        mf.mol.nelec = (2, 2)
        mf.mol.nao = 5
        eris = mock.Mock()
        eris.Loo = rng.standard_normal((5, 2, 2))
        eris.Lov = rng.standard_normal((5, 2, 3))
        eris.Lvo = eris.Lov.transpose(0, 2, 1).copy()
        eris.Lvv = rng.standard_normal((5, 3, 3))
        eris.foo = rng.standard_normal((2, 2))
        eris.fov = rng.standard_normal((2, 3))
        eris.fvv = rng.standard_normal((3, 3))
        eris.eia = 1.0 + rng.random((2, 3))
        eris.D = 1.0 + rng.random((2, 2, 3, 3))
        t1 = rng.standard_normal((2, 3))
        t2 = rng.standard_normal((2, 2, 3, 3))

        full = DFRPA(mf, eris, ladder_mode="full")
        active = DFRPA(mf, eris, act_particle=np.arange(3))
        full_update = full.updated_amps(t1.copy(), t2.copy())
        active_update = active.updated_amps(t1.copy(), t2.copy())

        np.testing.assert_allclose(active_update[0], full_update[0])
        np.testing.assert_allclose(active_update[1], full_update[1])
        np.testing.assert_allclose(active_update[2], full_update[2])


class FactoryMoCoefficientTest(unittest.TestCase):
    class FakeMF:
        @staticmethod
        def istype(method):
            return False

    def test_all_factories_forward_mo_coeff(self):
        mf = self.FakeMF()
        mo_coeff = object()
        eris = object()
        solver = object()
        cases = (
            (sylvester_cc.DFRCCSD, dfrccsd, "DFRCCSD"),
            (sylvester_cc.DFRPA, dfrpa, "DFRPA"),
            (
                sylvester_cc.DFRPASylvester,
                dfrpa_sylvester,
                "DFRPASylvester",
            ),
            (sylvester_cc.DFRPATriples, dfrpa_triples, "DFRPA"),
        )

        for factory, module, class_name in cases:
            with self.subTest(factory=factory.__name__):
                with mock.patch.object(
                    df_eri, "ERIs", return_value=eris
                ) as build_eris:
                    with mock.patch.object(
                        module, class_name, return_value=solver
                    ) as build_solver:
                        result = factory(mf, mo_coeff=mo_coeff)

                self.assertIs(result, solver)
                build_eris.assert_called_once_with(mf, mo_coeff=mo_coeff)
                build_solver.assert_called_once_with(mf, eris)

    def test_triples_factory_forwards_active_indices(self):
        mf = self.FakeMF()
        mo_coeff = object()
        eris = object()
        solver = object()
        act_hole = np.array([1, 2])
        act_particle = np.array([0, 3])

        with mock.patch.object(df_eri, "ERIs", return_value=eris):
            with mock.patch.object(
                dfrpa_triples, "DFRPA", return_value=solver
            ) as build_solver:
                result = sylvester_cc.DFRPATriples(
                    mf,
                    mo_coeff=mo_coeff,
                    act_hole=act_hole,
                    act_particle=act_particle,
                )

        self.assertIs(result, solver)
        build_solver.assert_called_once_with(
            mf,
            eris,
            act_hole=act_hole,
            act_particle=act_particle,
        )


if __name__ == "__main__":
    unittest.main()

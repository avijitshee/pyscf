def DFRCCSD(mf, mo_coeff=None, **kwargs):
    from pyscf.sylvester_cc import df_eri, dfrccsd

    if mf.istype('UHF'):
        raise RuntimeError('DFRCCSD cannot be used with UHF method.')

    eris = df_eri.ERIs(mf, mo_coeff=mo_coeff)
    return dfrccsd.DFRCCSD(mf, eris, **kwargs)


def DFRPA(mf, mo_coeff=None, **kwargs):
    from pyscf.sylvester_cc import df_eri, dfrpa

    if mf.istype('UHF'):
        raise RuntimeError('DFRPA cannot be used with UHF method.')

    eris = df_eri.ERIs(mf, mo_coeff=mo_coeff)
    return dfrpa.DFRPA(mf, eris, **kwargs)


def DFRPASylvester(mf, mo_coeff=None, **kwargs):
    from pyscf.sylvester_cc import df_eri, dfrpa_sylvester

    if mf.istype('UHF'):
        raise RuntimeError('DFRPASylvester cannot be used with UHF method.')

    eris = df_eri.ERIs(mf, mo_coeff=mo_coeff)
    return dfrpa_sylvester.DFRPASylvester(mf, eris, **kwargs)


def DFRPATriples(
    mf,
    mo_coeff=None,
    act_hole=None,
    act_particle=None,
    **kwargs,
):
    """Build the DFRPA variant containing the additional ladder diagram.

    The default active-``Jvv`` mode requires ``act_particle``.  Pass
    ``ladder_mode='full'`` only for exact reference calculations.
    """
    from pyscf.sylvester_cc import df_eri, dfrpa_triples

    if mf.istype('UHF'):
        raise RuntimeError('DFRPATriples cannot be used with UHF method.')

    eris = df_eri.ERIs(mf, mo_coeff=mo_coeff)
    if act_hole is not None:
        kwargs['act_hole'] = act_hole
    if act_particle is not None:
        kwargs['act_particle'] = act_particle
    return dfrpa_triples.DFRPA(mf, eris, **kwargs)

"""Finite-profile validation only; no JuTrack execution in these tests."""
import numpy as np
import pytest

from expert_modules.mba_design.periodic_hybrid import _profile_measurement


@pytest.mark.parametrize("field", ["Dx", "Dpx", "beta_x", "beta_y", "alpha_x", "s_m"])
def test_nonfinite_interior_profile_is_not_hidden_by_max(field):
    profile = [dict(s_m=s, Dx=.03, Dpx=0., beta_x=10., alpha_x=0., beta_y=10.)
               for s in np.linspace(0, 10, 101)]
    profile[50][field] = float("nan")
    with pytest.raises(ValueError, match="finite ordered profile"):
        _profile_measurement(dict(trace_x=.1, trace_y=.4, twiss_profile=profile),
                             dict(emittance_pm=40., energy_spread=.001),
                             dict(SF_L=(1.,2.), SD_L=(3.,4.), SD_R=(6.,7.), SF_R=(8.,9.)),
                             dict(s_start_m=4., s_end_m=6.))

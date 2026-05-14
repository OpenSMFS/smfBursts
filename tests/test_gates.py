from itertools import permutations

import numpy as np

import pytest

import smfbursts as smf


@pytest.fixture
def prd_p():
    return smf.Param(smf.Periods, {'detdef':smf.DetDef(2,2), 'period':60.0, 'start_at':'time_min', 'stop_at':'over'})


@pytest.fixture
def mr_pr_c(prd_p):
    return smf.Column(prd_p, 'max_rate', (smf.PhSel('0ex'), 10))

@pytest.fixture
def nph_raw(prd_p):
    return smf.Column(prd_p, 'nph_raw', (smf.PhSel('0ex')))

@pytest.fixture
def bva_c(prd_p):
    return smf.Column(prd_p, 'bva', (smf.PhSel('0ex0em'), smf.PhSel('0ex')))


@pytest.fixture
def g_nph10(nph_raw):
    try:
        return smf.make_geq_gate(nph_raw, 10.0)
    except:
        return None


@pytest.fixture
def g_nph20(nph_raw):
    try:
        return smf.make_geq_gate(nph_raw, 20.0)
    except:
        return None


@pytest.fixture
def g_mr50(mr_pr_c):
    try:
        return smf.make_geq_gate(mr_pr_c, 50.0)
    except:
        return None


@pytest.fixture
def g_mr100(mr_pr_c):
    try:
        return smf.make_geq_gate(mr_pr_c, 100.0)
    except:
        return None


@pytest.fixture
def g_bva1(bva_c):
    try:
        return smf.make_geq_gate(bva_c, 0.1)
    except:
        return None


@pytest.fixture
def g_bva2(bva_c):
    try:
        return smf.make_geq_gate(bva_c, 0.2)
    except:
        return None


@pytest.fixture
def bg_p(prd_p):
    return smf.Param(smf.BG, {'func':smf.bg.exp_mlehist, 'tail_min':4e-4, 'compute_stream':'single_all'}, {'base':prd_p})


@pytest.fixture
def brst_p(bg_p):
    return smf.Param(smf.Bursts, {'streams':(smf.PhSel('0ex'),), 'm':np.array([10]), 'F':np.array([6.0])}, {'bg':bg_p})


@pytest.mark.dependency(name='make_geq_gate')
def test_make_geq_gate(nph_raw):
    g0 = smf.make_geq_gate(nph_raw, 10.0)
    gate = smf.Gate(smf.gates.LIN_GEQ_gate, (nph_raw, ), {'m':10.0, 'vec':np.array([1.0])})
    gg = smf.GateGroup(np.array([False, True]), gate)
    assert g0 == gg
    assert g0 in gg
    assert gg in g0


def test_make_lt_gate(nph_raw):
    g0 = smf.make_lt_gate(nph_raw, 10.0, exclude_nan=False)
    gate = smf.Gate(smf.gates.LIN_GEQ_gate, (nph_raw, ), {'m':10.0, 'vec':np.array([1.0])})
    gg = smf.GateGroup(np.array([True, False]), gate)
    assert g0 == gg
    assert g0 in gg
    assert gg in g0


@pytest.mark.dependency(name='singlegate')
def test_single_gate_overlap_g_ng(g_nph10):
    assert smf.GateGroup.overlap(g_nph10, ~g_nph10) == 0b0110, "Incorrect overlap g0 ~g0"
    assert smf.GateGroup.overlap(~g_nph10, g_nph10) == 0b0110, "Incorrect overlap g0 ~g0"


@pytest.mark.dependency(name='singlegate')
def test_single_gate_overlap_gb_in_ga(g_nph10, g_nph20):
    assert smf.GateGroup.overlap(g_nph10, g_nph20) == 0b1011, "Incorect overlap g0 g1"
    assert smf.GateGroup.overlap(g_nph20, g_nph10) == 0b1101, "Incorect overlap g1 g0 (reverse)"


@pytest.mark.dependency(name='singlegate')
def test_single_gate_overlap_gb_in_ga_use_nbg(g_nph10, g_nph20):
    assert smf.GateGroup.overlap(~g_nph20, g_nph10) == 0b1110, "Incorrect geq-le overlap ~g1 g0"
    assert smf.GateGroup.overlap(g_nph10, ~g_nph20) == 0b1110, "Incorrect geq-le overlap g0 ~g1, (reverse)"


@pytest.mark.dependency(name='singlegate')
def test_single_gate_overlap_gb_in_ga_use_nba(g_nph10, g_nph20):
    assert smf.GateGroup.overlap(~g_nph10, g_nph20) == 0b0111, "Incorrect geq-le overlap ~g0 g1"
    assert smf.GateGroup.overlap(g_nph20, ~g_nph10) == 0b0111, "Incorrect geq-le overlap ~g0 g1"


def gate_and(*args:smf.GateGroup)->smf.GateGroup:
    out = args[0]
    for arg in args[1:]:
        out &= arg
    return out


def gate_or(*args:smf.GateGroup)->smf.GateGroup:
    out = args[0]
    for arg in args[1:]:
        out |= arg
    return out


def gate_eq(*args:smf.GateGroup)->smf.GateGroup:
    out = args[0]
    for arg in args[1:]:
        out @= arg
    return out


def gate_xor(*args:smf.GateGroup)->smf.GateGroup:
    out = args[0]
    for arg in args[1:]:
        out ^= arg
    return out


def gate_sub(a:smf.GateGroup, b:smf.GateGroup())->smf.GateGroup:
    return a - b


def gate_implies(a:smf.GateGroup, b:smf.GateGroup())->smf.GateGroup:
    return a >> b


def gate_rimplies(a:smf.GateGroup, b:smf.GateGroup())->smf.GateGroup:
    return a << b


@pytest.fixture(params=[gate_and, gate_or, gate_eq, gate_xor])
def gate_commutative(request):
    return request.param


@pytest.fixture(params=[gate_sub, gate_implies, gate_rimplies])
def gate_noncommutative(request):
    return request.param



@pytest.mark.dependency(name='commutativegate')
def test_commutative(g_nph10, g_mr100, gate_commutative):
    g_n_mr = gate_commutative(g_nph10, g_mr100)
    assert gate_commutative(g_mr100,g_nph10) == g_n_mr, "operation not commutative"
    assert g_n_mr.truthtable.ndim == 2, "Wrong dimensions of table"


@pytest.mark.dependency(name='noncommutativegate')
def test_noncommutative(g_nph10, g_mr100, gate_noncommutative):
    g_n_mr = gate_noncommutative(g_nph10, g_mr100)
    assert gate_noncommutative(g_mr100, g_nph10) != g_n_mr, "non-commutative incorrectlry evaluates as commutative"
    assert g_n_mr.truthtable.ndim == 2, "Wrong dimensions of table"


@pytest.mark.dependency(depends=['noncommutativegate',])
def test_subtraction_equivalence(g_nph10, g_mr100):
    assert g_nph10 - g_mr100 == g_nph10 & ~g_mr100, 'Subtraction incorrectly defined, not equivalent to A & ~B'
    assert g_mr100 - g_nph10 == g_mr100 & ~g_nph10, 'Subtraction incorrectly defined, not equivalent to A & ~B'


@pytest.mark.dependency(depends=['noncommutativegate',])
def test_implies_rimplies_equivalence(g_nph10, g_mr100):
    assert g_nph10 >> g_mr100 == g_mr100 << g_nph10, 'Incorrect implication/reverse implication evaluation :: A>>B != B<<A'


@pytest.mark.dependency(depends=['commutativegate',])
def test_AND_reduction(g_nph10, g_nph20):
    assert (g_nph10 & ~g_nph10).truthtable.ndim == 0, "Failed reduction to FALSE gate of A & ~A"
    assert g_nph10 & g_nph20 == g_nph20, "Failed reduction of >10 & >20 -> >20"
    assert ~g_nph10 & ~g_nph20 == ~g_nph10, "Failed reduction of <10 & <20 to <10"
    g_nph_rng = g_nph10 & ~g_nph20
    assert g_nph_rng.truthtable.ndim == 2, 'reduced to incorrect truthtable dimension'
    assert g_nph_rng.truthtable.sum() == 1, "Too many True values in truthtable"
    p_true = tuple(1 if g == g_nph10.gates[0] else 0 for g in g_nph_rng.gates)
    assert g_nph_rng.truthtable[p_true], "Incorrect position of True for range gate"
    

@pytest.mark.dependency(depends=['commutativegate',])
def test_OR_reduction(g_nph10, g_nph20):
    assert g_nph10 | g_nph20 == g_nph10, "Failed OR reduction of >10 | >20 to >10"
    assert ~g_nph10 | ~g_nph20 == ~g_nph20, "Failed OR reduction of <10 | < 20 to <20"
    assert ~g_nph20 | ~g_nph10 == ~g_nph20, "Failed OR reduction of <10 | < 20 to <20"
    g_or = ~g_nph10 | g_nph20
    assert g_or.truthtable.ndim == 2, "<10 | >20 incorrect truthtable size"
    assert not np.all(g_or.truthtable), "Incorrect number of True in truthtable of OR"
    assert g_or.truthtable[0,0] and g_or.truthtable[1,1], "Incorrect placement of OR during reduction"
    p_false = tuple(1 if g == g_nph10.gates[0] else 0 for g in g_or.gates)
    assert not g_or.truthtable[p_false], "Wrong location of False in or gate"


@pytest.mark.dependency(depends=['commutativegate',])
def test_3way_commutative(g_nph10, g_mr100, g_bva1, gate_commutative):
    g_com = gate_commutative(g_nph10, g_mr100, g_bva1)
    assert g_com.truthtable.ndim == 3, "3 way and has incorrect number of dimensions"
    if gate_commutative == gate_and:
        assert g_com.truthtable.sum() == 1, "Too many True for all AND gate"
        assert g_com.truthtable[1,1,1], "Wrong location of True in all AND gate"
    elif gate_commutative == gate_or:
        assert g_com.truthtable.sum() == 7, "Wrong number of True for all OR gate"
        assert not g_com.truthtable[0,0,0], "Wrong location of FALSE for all OR gate"
    elif gate_commutative in (gate_eq, gate_xor):
        assert np.all(g_com.truthtable == np.array([[[False, True],[True, False]], [[True, False], [False, True]]])), "Wrong 3 way EQ table"
    for a, b, c in permutations((g_nph10, g_mr100, g_bva1), 3):
        assert gate_commutative(a, b, c) == g_com, "3 way AND non-commutative"
        if gate_commutative == gate_and:
            assert smf.GateGroup.overlap(g_com, a) == 0b1101
            assert smf.GateGroup.overlap(g_com, b) == 0b1101
            assert smf.GateGroup.overlap(g_com, c) == 0b1101
            assert smf.GateGroup.overlap(a, g_com) == 0b1011
            assert smf.GateGroup.overlap(b, g_com) == 0b1011
            assert smf.GateGroup.overlap(c, g_com) == 0b1011
        elif gate_commutative == gate_or:
            assert smf.GateGroup.overlap(g_com, a) == 0b1011
            assert smf.GateGroup.overlap(g_com, b) == 0b1011
            assert smf.GateGroup.overlap(g_com, c) == 0b1011
            assert smf.GateGroup.overlap(a, g_com) == 0b1101
            assert smf.GateGroup.overlap(b, g_com) == 0b1101
            assert smf.GateGroup.overlap(c, g_com) == 0b1101


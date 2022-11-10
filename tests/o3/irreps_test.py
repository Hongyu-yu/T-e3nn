import pytest

from e3nn import o3
from e3nn.o3 import irrep


def test_creation():
    o3.Irrep(3, 1)
    ir = o3.Irrep("3e")
    o3.Irrep(ir)
    assert o3.Irrep("10o") == o3.Irrep(10, -1)

    irreps = o3.Irreps(ir)
    o3.Irreps(irreps)
    o3.Irreps([(32, (4, -1))])
    o3.Irreps("11e")
    assert o3.Irreps("16x1e + 32 x 2o") == o3.Irreps([(16, (1, 1)), (32, (2, -1))])
    o3.Irreps(["1e", "2o"])
    o3.Irreps([(16, "3e"), "1e"])
    o3.Irreps([(16, "3e"), "1e", (256, (1, -1))])

    assert irrep.l0e == o3.Irrep("0e")
    from e3nn.o3.irrep import l1o

    assert l1o == o3.Irrep("1o")


def test_time_reversal_creation():
    o3.Irrep(3, 1, -1)
    ir = o3.Irrep("3eo")
    o3.Irrep(ir)
    assert o3.Irrep("10oo") == o3.Irrep(10, -1, -1)

    irreps = o3.Irreps(ir)
    o3.Irreps(irreps)
    o3.Irreps([(32, (4, -1, -1))])
    o3.Irreps("11eo")
    assert o3.Irreps("16x1eo + 32 x 2oo") == o3.Irreps([(16, (1, 1, -1)), (32, (2, -1, -1))])
    o3.Irreps(["1eo", "2oo"])
    o3.Irreps([(16, "3eo"), "1eo"])
    o3.Irreps([(16, "3eo"), "1eo", (256, (1, -1, -1))])

    assert irrep.l0eo == o3.Irrep("0eo")
    from e3nn.o3.irrep import l1oo

    assert l1oo == o3.Irrep("1oo")


def test_properties():
    irrep = o3.Irrep("3e")
    assert irrep.l == 3
    assert irrep.p == 1
    assert irrep.dim == 7

    assert o3.Irrep(repr(irrep)) == irrep

    l, p, t = o3.Irrep("5o")
    assert l == 5
    assert p == -1
    assert t == 1

    iterator = o3.Irrep.iterator(5)
    assert len(list(iterator)) == 24

    iterator = o3.Irrep.iterator()
    for x in range(100):
        irrep = next(iterator)
        assert irrep.l == x // 4
        assert irrep.p in (-1, 1)
        assert irrep.t in (-1, 1)
        assert irrep.dim == 2 * (x // 4) + 1

    irreps = o3.Irreps("4x1e + 6x2e + 12x2o")
    assert o3.Irreps(repr(irreps)) == irreps


def test_time_reversal_properties():
    irrep = o3.Irrep("3eo")
    assert irrep.l == 3
    assert irrep.p == 1
    assert irrep.t == -1
    assert irrep.dim == 7

    assert o3.Irrep(repr(irrep)) == irrep

    l, p, t = o3.Irrep("5oo")
    assert l == 5
    assert p == -1
    assert t == -1

    irreps = o3.Irreps("4x1eo + 6x2eo + 12x2oe")
    assert o3.Irreps(repr(irreps)) == irreps


def test_arithmetic():
    assert 3 * o3.Irrep("6o") == o3.Irreps("3x6o")
    products = list(o3.Irrep("1o") * o3.Irrep("2e"))
    assert products == [o3.Irrep("1o"), o3.Irrep("2o"), o3.Irrep("3o")]

    assert o3.Irrep("4o") + o3.Irrep("7e") == o3.Irreps("4o + 7e")

    assert 2 * o3.Irreps("2x2e + 4x1o") == o3.Irreps("2x2e + 4x1o + 2x2e + 4x1o")
    assert o3.Irreps("2x2e + 4x1o") * 2 == o3.Irreps("2x2e + 4x1o + 2x2e + 4x1o")

    assert o3.Irreps("1o + 4o") + o3.Irreps("1o + 7e") == o3.Irreps("1o + 4o + 1o + 7e")


def test_time_reversal_arithmetic():
    assert 3 * o3.Irrep("6oo") == o3.Irreps("3x6oo")
    products = list(o3.Irrep("1oe") * o3.Irrep("2eo"))
    assert products == [o3.Irrep("1oo"), o3.Irrep("2oo"), o3.Irrep("3oo")]

    assert o3.Irrep("4oo") + o3.Irrep("7ee") == o3.Irreps("4oo + 7ee")

    assert 2 * o3.Irreps("2x2eo + 4x1oe") == o3.Irreps("2x2eo + 4x1oe + 2x2eo + 4x1oe")
    assert o3.Irreps("2x2ee + 4x1oo") * 2 == o3.Irreps("2x2ee + 4x1oo + 2x2ee + 4x1oo")

    assert o3.Irreps("1oe + 4oe") + o3.Irreps("1oo + 7eo") == o3.Irreps("1oe + 4oe + 1oo + 7eo")


def test_empty_irreps():
    assert o3.Irreps() == o3.Irreps("") == o3.Irreps([])
    assert len(o3.Irreps()) == 0
    assert o3.Irreps().dim == 0
    assert o3.Irreps().ls == []
    assert o3.Irreps().num_irreps == 0


def test_getitem():
    irreps = o3.Irreps("16x1e + 3e + 2e + 5o")
    assert irreps[0] == (16, o3.Irrep("1e"))
    assert irreps[3] == (1, o3.Irrep("5o"))
    assert irreps[-1] == (1, o3.Irrep("5o"))

    sliced = irreps[2:]
    assert isinstance(sliced, o3.Irreps)
    assert sliced == o3.Irreps("2e + 5o")


def test_cat():
    irreps = o3.Irreps("4x1e + 6x2e + 12x2o") + o3.Irreps("1x1e + 2x2e + 12x4o")
    assert len(irreps) == 6
    assert irreps.ls == [1] * 4 + [2] * 6 + [2] * 12 + [1] * 1 + [2] * 2 + [4] * 12
    assert irreps.lmax == 4
    assert irreps.num_irreps == 4 + 6 + 12 + 1 + 2 + 12


def test_contains():
    assert o3.Irrep("2e") in o3.Irreps("3x0e + 2x2e + 1x3o")
    assert o3.Irrep("2o") not in o3.Irreps("3x0e + 2x2e + 1x3o")


def test_errors():
    """Test invalid irrep specifications"""
    # Irrep
    with pytest.raises(ValueError):
        o3.Irrep(-1)

    with pytest.raises(ValueError):
        o3.Irrep(1, p=2)

    with pytest.raises(ValueError):
        o3.Irrep("-1e")

    # Irreps
    with pytest.raises(ValueError):
        o3.Irreps("-1x1e")

    with pytest.raises(ValueError):
        o3.Irreps("1x-1e")

    with pytest.raises(ValueError):
        o3.Irreps("bla")


@pytest.mark.xfail()
def test_fail1():
    o3.Irreps([(32, 1)])

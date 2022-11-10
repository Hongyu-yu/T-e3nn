import torch

from e3nn import o3
from e3nn.nn import Identity
from e3nn.o3 import FullyConnectedTensorProduct, FullTensorProduct, Norm, TensorSquare
from e3nn.util.test import assert_equivariant, assert_auto_jitable


def test_fully_connected():
    irreps_in1 = o3.Irreps("1e + 2e + 3x3o + 1eo + 2eo + 3x3oo")
    irreps_in2 = o3.Irreps("1e + 2e + 3x3o + 1eo + 2eo + 3x3oo")
    irreps_out = o3.Irreps("1e + 2e + 3x3o + 1eo + 2eo + 3x3oo")
    m = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out)
    print(m)
    m(torch.randn(irreps_in1.dim), torch.randn(irreps_in2.dim))

    assert_equivariant(m)
    assert_auto_jitable(m)


def test_fully_connected_normalization():
    m = FullyConnectedTensorProduct("10x0e", "10x0e", "0e")
    for p in m.parameters():
        p.data.fill_(1.0)

    n = FullyConnectedTensorProduct("3x0e + 7x0e", "3x0e + 7x0e", "0e")
    for p in n.parameters():
        p.data.fill_(1.0)

    x1, x2 = torch.randn(2, 3, 10)
    assert torch.allclose(m(x1, x2), n(x1, x2))


def test_id():
    irreps_in = o3.Irreps("1e + 2eo + 3x3o")
    irreps_out = o3.Irreps("1e + 2eo + 3x3o")
    m = Identity(irreps_in, irreps_out)
    print(m)
    m(torch.randn(irreps_in.dim))

    assert_equivariant(m)
    assert_auto_jitable(m, strict_shapes=False)


def test_full():
    irreps_in1 = o3.Irreps("1eo + 2ee + 3x3oo")
    irreps_in2 = o3.Irreps("1ee + 2x2eo + 2x3oe")
    m = FullTensorProduct(irreps_in1, irreps_in2)
    print(m)

    assert_equivariant(m)
    assert_auto_jitable(m)


def test_norm():
    irreps_in = o3.Irreps("3x0e + 5x1o + 3x0eo + 5x1oo")
    scalars = torch.randn(3)
    vecs = torch.randn(5, 3)
    scalars_tr = torch.randn(3)
    vecs_tr = torch.randn(5, 3)
    norm = Norm(irreps_in=irreps_in)
    out_norms = norm(
        torch.cat((scalars.reshape(1, -1), vecs.reshape(1, -1), scalars_tr.reshape(1, -1), vecs_tr.reshape(1, -1)), dim=-1)
    )
    true_scalar_norms = torch.abs(scalars)
    true_vec_norms = torch.linalg.norm(vecs, dim=-1)
    true_scalar_tr_norms = torch.abs(scalars_tr)
    true_vec_tr_norms = torch.linalg.norm(vecs_tr, dim=-1)
    assert torch.allclose(out_norms[0, :3], true_scalar_norms)
    assert torch.allclose(out_norms[0, 3:8], true_vec_norms)
    assert torch.allclose(out_norms[0, 8:11], true_scalar_tr_norms)
    assert torch.allclose(out_norms[0, 11:], true_vec_tr_norms)

    assert_equivariant(norm)
    assert_auto_jitable(norm)


def test_square_normalization():
    irreps = o3.Irreps("0e + 1e + 2e + 0eo + 1eo + 2eo")
    tp = TensorSquare(irreps, irrep_normalization="norm")
    x = irreps.randn(1_000_000, -1, normalization="norm")
    y = tp(x)
    n = Norm(tp.irreps_out, squared=True)(y)

    assert (n.mean(0).log().abs().exp() < 1.1).all()

    irreps = o3.Irreps("0e + 3x1e + 3e + 0eo + 3x1eo + 3eo")
    tp = o3.TensorSquare(irreps, irrep_normalization="component")
    x = irreps.randn(1_000_000, -1, normalization="component")
    y = tp(x)

    assert (y.pow(2).mean(0).log().abs().exp() < 1.1).all()

    tp = TensorSquare(irreps, irrep_normalization="none")
    y = tp(x)

    assert not (y.pow(2).mean(0).log().abs().exp() < 1.1).all()

    # with weights
    tp = TensorSquare(irreps, irreps)

    n = 2_000
    y = torch.stack([tp(tp.irreps_in.randn(n, -1), torch.randn(tp.weight_numel)) for _ in range(n)])

    assert (y.pow(2).mean([0, 1]).log().abs().exp() < 1.1).all()


def test_square_elasticity_tensor():
    tp = TensorSquare("1oo")
    tp = TensorSquare(tp.irreps_out)  # "2e,"
    assert tp.irreps_out.simplify() == o3.Irreps("2x0e + 2x2e + 4e")

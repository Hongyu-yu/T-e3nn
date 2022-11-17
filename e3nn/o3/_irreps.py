import itertools
import collections
from typing import List, Union

import torch

from e3nn.math import direct_sum, perm

# These imports avoid cyclic reference from o3 itself
from . import _rotation
from . import _wigner


class Irrep(tuple):
    r"""Irreducible representation of :math:`O(3)` -> 'U(2)'

    This class does not contain any data, it is a structure that describe the representation.
    It is typically used as argument of other classes of the library to define the input and output representations of
    functions.

    Parameters
    ----------
    l : int
        non-negative integer, the degree of the representation, :math:`l = 0, 1, \dots`

    p : {1, -1} -> {e, o}
        the parity of the representation

    t : {1, -1} -> {e, o}
        the time reversal of the representation

    Examples
    --------
    Create a scalar representation (:math:`l=0`) of even parity and even T.

    >>> Irrep(0, 1, 1)
    0ee

    Create a pseudotensor representation (:math:`l=2`) of odd parity and even T.

    >>> Irrep(2, -1, 1)
    2oe

    >>> Irrep("2o").dim
    5

    >>> Irrep("2ee") in Irrep("1oe") * Irrep("1oe")
    True

    >>> Irrep("1oe") + Irrep("2oe")
    1x1oe+1x2oe
    """

    def __new__(cls, l: Union[float, "Irrep", str, tuple], p=None, t=1):
        if p is None:
            if isinstance(l, Irrep):
                return l

            if isinstance(l, str):
                try:
                    name = l.strip()
                    if name[-2] not in ["e", "o", "y"]:
                        l = int(name[:-1])
                        ind_dict: dict = {"e": 1, "o": -1, "y": (-1) ** l}
                        p = ind_dict[name[-1]]
                        t = 1  # Default t is 1
                    else:
                        l = int(name[:-2])
                        ind_dict: dict = {"e": 1, "o": -1, "y": (-1) ** l}
                        p = ind_dict[name[-2]]
                        t = ind_dict[name[-1]]
                    assert l >= 0
                except Exception:
                    raise ValueError(f'unable to convert string "{name}" into an Irrep')
            elif isinstance(l, tuple):
                if len(l) == 2:
                    l, p = l
                    t = 1
                elif len(l) == 3:
                    l, p, t = l
                else:
                    raise ValueError(f'unable to convert tuple "{l}" into an Irrep. It should consist of 3 or 2 integers')

        if not isinstance(l, int) or l < 0:
            raise ValueError(f"l must be positive integer, got {l}")
        if p not in (-1, 1):
            raise ValueError(f"parity must be on of (-1, 1), got {p}")
        if t not in (-1, 1):
            raise ValueError(f"time reversal must be on of (-1, 1), got {t}")
        return super().__new__(cls, (l, p, t))

    @property
    def l(self) -> int:  # noqa: E743
        r"""The degree of the representation, :math:`l = 0, 1, \dots`."""
        return self[0]

    @property
    def p(self) -> int:
        r"""The parity of the representation, :math:`p = \pm 1`."""
        return self[1]

    @property
    def t(self) -> int:
        r"""The time reversal of the representation, :math:`p = \pm 1`."""
        return self[2]

    def __repr__(self):
        p = {+1: "e", -1: "o"}[self.p]
        t = {+1: "e", -1: "o"}[self.t]
        return f"{self.l}{p}{t}"

    @classmethod
    def iterator(cls, lmax=None):
        r"""Iterator through all the irreps of :math:`O(3)`

        Examples
        --------
        >>> it = Irrep.iterator()
        >>> next(it), next(it), next(it), next(it)
        (0e, 0o, 1o, 1e)
        """
        for l in itertools.count():
            yield Irrep(l, (-1) ** l, (-1) ** l)
            yield Irrep(l, (-1) ** l, -((-1) ** l))
            yield Irrep(l, -((-1) ** l), (-1) ** l)
            yield Irrep(l, -((-1) ** l), -((-1) ** l))

            if l == lmax:
                break

    def D_from_angles(self, alpha, beta, gamma, k=None, kt=None):
        r"""Matrix :math:`p^k D^l(\alpha, \beta, \gamma)`

        (matrix) Representation of :math:`O(3)`. :math:`D` is the representation of :math:`SO(3)`, see `wigner_D`.

        Parameters
        ----------
        alpha : `torch.Tensor`
            tensor of shape :math:`(...)`
            Rotation :math:`\alpha` around Y axis, applied third.

        beta : `torch.Tensor`
            tensor of shape :math:`(...)`
            Rotation :math:`\beta` around X axis, applied second.

        gamma : `torch.Tensor`
            tensor of shape :math:`(...)`
            Rotation :math:`\gamma` around Y axis, applied first.

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`
            How many times the parity is applied.

        kt : `torch.Tensor`, optional
            tensor of shape :math:`(...)`
            How many times the time reversal operation is applied.

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., 2l+1, 2l+1)`

        See Also
        --------
        o3.wigner_D
        Irreps.D_from_angles
        """
        if k is None:
            k = torch.zeros_like(alpha)
        if kt is None:
            kt = torch.zeros_like(alpha)

        alpha, beta, gamma, k, kt = torch.broadcast_tensors(alpha, beta, gamma, k, kt)
        return _wigner.wigner_D(self.l, alpha, beta, gamma) * self.p ** k[..., None, None] * self.t ** kt[..., None, None]

    def D_from_quaternion(self, q, k=None, kt=None):
        r"""Matrix of the representation, see `Irrep.D_from_angles`

        Parameters
        ----------
        q : `torch.Tensor`
            tensor of shape :math:`(..., 4)`

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        kt : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., 2l+1, 2l+1)`
        """
        return self.D_from_angles(*_rotation.quaternion_to_angles(q), k, kt)

    def D_from_matrix(self, R, parity=True, time_reversal=False):
        r"""Matrix of the representation, see `Irrep.D_from_angles`

        Determination of the Matrix: (k in the func) 1: do reverse; 0: remain the same

        Parameters
        ----------
        R : `torch.Tensor`
            tensor of shape :math:`(..., 3, 3)`

        parity: `bool`
            matrix consider parity

        time_reversal: `bool`
            matrix consider time_reversal

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., 2l+1, 2l+1)`

        Examples
        --------
        >>> m = Irrep(1, -1).D_from_matrix(-torch.eye(3))
        >>> m.long()
        tensor([[-1,  0,  0],
                [ 0, -1,  0],
                [ 0,  0, -1]])
        """
        d = torch.det(R).sign()
        R = d[..., None, None] * R
        k = (1 - d) / 2
        kp = k if parity else torch.as_tensor(0).type_as(k)
        kt = k if time_reversal else torch.as_tensor(0).type_as(k)
        return self.D_from_angles(*_rotation.matrix_to_angles(R), kp, kt)

    def D_from_axis_angle(self, axis, angle):
        r"""Matrix of the representation, see `Irrep.D_from_angles`

        Parameters
        ----------
        axis : `torch.Tensor`
            tensor of shape :math:`(..., 3)`

        angle : `torch.Tensor`
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., 2l+1, 2l+1)`
        """
        return self.D_from_angles(*_rotation.axis_angle_to_angles(axis, angle))

    @property
    def dim(self) -> int:
        """The dimension of the representation, :math:`2 l + 1`."""
        return 2 * self.l + 1

    def is_scalar(self) -> bool:
        """Equivalent to ``l == 0 and p == 1 and t==1``"""
        return self.l == 0 and self.p == 1 and self.t == 1

    def __mul__(self, other):
        r"""Generate the irreps from the product of two irreps.

        Returns
        -------
        generator of `e3nn.o3.Irrep`
        """
        other = Irrep(other)
        p = self.p * other.p
        t = self.t * other.t
        lmin = abs(self.l - other.l)
        lmax = self.l + other.l
        for l in range(lmin, lmax + 1):
            yield Irrep(l, p, t)

    def count(self, _value):
        raise NotImplementedError

    def index(self, _value):
        raise NotImplementedError

    def __rmul__(self, other):
        r"""
        >>> 3 * Irrep('1e')
        3x1ee
        """
        assert isinstance(other, int)
        return Irreps([(other, self)])

    def __add__(self, other):
        return Irreps(self) + Irreps(other)

    def __contains__(self, _object):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class _MulIr(tuple):
    def __new__(cls, mul, ir=None):
        if ir is None:
            mul, ir = mul

        assert isinstance(mul, int)
        assert isinstance(ir, Irrep)
        return super().__new__(cls, (mul, ir))

    @property
    def mul(self) -> int:
        return self[0]

    @property
    def ir(self) -> Irrep:
        return self[1]

    @property
    def dim(self) -> int:
        return self.mul * self.ir.dim

    def __repr__(self):
        return f"{self.mul}x{self.ir}"

    def __getitem__(self, item) -> Union[int, Irrep]:  # pylint: disable=useless-super-delegation
        return super().__getitem__(item)

    def count(self, _value):
        raise NotImplementedError

    def index(self, _value):
        raise NotImplementedError


class Irreps(tuple):
    r"""Direct sum of irreducible representations of :math:`O(3)`

    This class does not contain any data, it is a structure that describe the representation.
    It is typically used as argument of other classes of the library to define the input and output representations of
    functions.

    Attributes
    ----------
    dim : int
        the total dimension of the representation

    num_irreps : int
        number of irreps. the sum of the multiplicities

    ls : list of int
        list of :math:`l` values

    lmax : int
        maximum :math:`l` value

    Examples
    --------
    Create a representation of 100 :math:`l=0` of even parity and 50 pseudo-vectors.

    >>> x = Irreps([(100, (0, 1, 1)), (50, (1, 1, 1))])
    >>> x
    100x0ee+50x1ee

    >>> x.dim
    250

    Create a representation of 100 :math:`l=0` of even parity and 50 pseudo-vectors.

    >>> Irreps("100x0ee + 50x1ee")
    100x0ee+50x1ee

    >>> Irreps("100x0ee + 50x1ee + 0x2ee")
    100x0ee+50x1ee+0x2ee

    >>> Irreps("100x0e + 50x1e + 0x2e").lmax
    1

    >>> Irrep("2ee") in Irreps("0ee + 2ee")
    True

    Empty Irreps

    >>> Irreps(), Irreps("")
    (, )
    """

    def __new__(cls, irreps=None) -> Union[_MulIr, "Irreps"]:
        if isinstance(irreps, Irreps):
            return super().__new__(cls, irreps)

        out = []
        if isinstance(irreps, Irrep):
            out.append(_MulIr(1, Irrep(irreps)))
        elif isinstance(irreps, str):
            try:
                if irreps.strip() != "":
                    for mul_ir in irreps.split("+"):
                        if "x" in mul_ir:
                            mul, ir = mul_ir.split("x")
                            mul = int(mul)
                            ir = Irrep(ir)
                        else:
                            mul = 1
                            ir = Irrep(mul_ir)

                        assert isinstance(mul, int) and mul >= 0
                        out.append(_MulIr(mul, ir))
            except Exception:
                raise ValueError(f'Unable to convert string "{irreps}" into an Irreps')
        elif irreps is None:
            pass
        else:
            for mul_ir in irreps:
                mul = None
                ir = None

                if isinstance(mul_ir, str):
                    mul = 1
                    ir = Irrep(mul_ir)
                elif isinstance(mul_ir, Irrep):
                    mul = 1
                    ir = mul_ir
                elif isinstance(mul_ir, _MulIr):
                    mul, ir = mul_ir
                elif len(mul_ir) == 2:
                    mul, ir = mul_ir
                    ir = Irrep(ir)

                if not (isinstance(mul, int) and mul >= 0 and ir is not None):
                    raise ValueError(f'Unable to interpret "{mul_ir}" as an irrep.')

                out.append(_MulIr(mul, ir))
        return super().__new__(cls, out)

    @staticmethod
    def spherical_harmonics(lmax, p=-1, t=1):
        r"""representation of the spherical harmonics

        Parameters
        ----------
        lmax : int
            maximum :math:`l`

        p : {1, -1}
            the parity of the representation

        t : {1, -1}
            the time reversal of the representation

        Returns
        -------
        `e3nn.o3.Irreps`
            representation of :math:`(Y^0, Y^1, \dots, Y^{\mathrm{lmax}})`

        Examples
        --------

        1) For rij (Default)
        >>> Irreps.spherical_harmonics(3)
        1x0ee+1x1oe+1x2ee+1x3oe

        2) For si
        >>> Irreps.spherical_harmonics(3, p=1, t=-1)
        1x0ee+1x1eo+1x2ee+1x3eo

        3) For si X sj, si*sj
        >>> Irreps.spherical_harmonics(4, p=1, t=1)
        1x0ee+1x1ee+1x2ee+1x3ee+1x4ee

        4) For rij X si
        >>> Irreps.spherical_harmonics(4, p=-1, t=-1)
        1x0ee+1x1oo+1x2ee+1x3oo
        """
        return Irreps([(1, (l, p ** l, t ** l)) for l in range(lmax + 1)])

    def slices(self):
        r"""List of slices corresponding to indices for each irrep.

        Examples
        --------

        >>> Irreps('2x0e + 1e').slices()
        [slice(0, 2, None), slice(2, 5, None)]
        """
        s = []
        i = 0
        for mul_ir in self:
            s.append(slice(i, i + mul_ir.dim))
            i += mul_ir.dim
        return s

    def randn(self, *size, normalization="component", requires_grad=False, dtype=None, device=None):
        r"""Random tensor.

        Parameters
        ----------
        *size : list of int
            size of the output tensor, needs to contains a ``-1``

        normalization : {'component', 'norm'}

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``size`` where ``-1`` is replaced by ``self.dim``

        Examples
        --------

        >>> Irreps("5x0ee + 10x1oe").randn(5, -1, 5, normalization='norm').shape
        torch.Size([5, 35, 5])

        >>> random_tensor = Irreps("2oe").randn(2, -1, 3, normalization='norm')
        >>> random_tensor.norm(dim=1).sub(1).abs().max().item() < 1e-5
        True
        """
        di = size.index(-1)
        lsize = size[:di]
        rsize = size[di + 1 :]

        if normalization == "component":
            return torch.randn(*lsize, self.dim, *rsize, requires_grad=requires_grad, dtype=dtype, device=device)
        elif normalization == "norm":
            x = torch.zeros(*lsize, self.dim, *rsize, requires_grad=requires_grad, dtype=dtype, device=device)
            with torch.no_grad():
                for s, (mul, ir) in zip(self.slices(), self):
                    r = torch.randn(*lsize, mul, ir.dim, *rsize, dtype=dtype, device=device)
                    r.div_(r.norm(2, dim=di + 1, keepdim=True))
                    x.narrow(di, s.start, mul * ir.dim).copy_(r.reshape(*lsize, -1, *rsize))
            return x
        else:
            raise ValueError("Normalization needs to be 'norm' or 'component'")

    def __getitem__(self, i) -> Union[_MulIr, "Irreps"]:
        x = super().__getitem__(i)
        if isinstance(i, slice):
            return Irreps(x)
        return x

    def __contains__(self, ir) -> bool:
        ir = Irrep(ir)
        return ir in (irrep for _, irrep in self)

    def count(self, ir) -> int:
        r"""Multiplicity of ``ir``.

        Parameters
        ----------
        ir : `e3nn.o3.Irrep`

        Returns
        -------
        `int`
            total multiplicity of ``ir``
        """
        ir = Irrep(ir)
        return sum(mul for mul, irrep in self if ir == irrep)

    def index(self, _object):
        raise NotImplementedError

    def __add__(self, irreps):
        irreps = Irreps(irreps)
        return Irreps(super().__add__(irreps))

    def __mul__(self, other):
        r"""
        >>> (Irreps('2x1e') * 3).simplify()
        6x1e
        """
        if isinstance(other, Irreps):
            raise NotImplementedError("Use o3.TensorProduct for this, see the documentation")
        return Irreps(super().__mul__(other))

    def __rmul__(self, other):
        r"""
        >>> 2 * Irreps('0e + 1e')
        1x0e+1x1e+1x0e+1x1e
        """
        return Irreps(super().__rmul__(other))

    def simplify(self) -> "Irreps":
        """Simplify the representations.

        Returns
        -------
        `e3nn.o3.Irreps`

        Examples
        --------

        Note that simplify does not sort the representations.

        >>> Irreps("1ee + 1ee + 0ee").simplify()
        2x1ee+1x0ee

        Equivalent representations which are separated from each other are not combined.

        >>> Irreps("1ee + 1ee + 0ee + 1ee").simplify()
        2x1ee+1x0ee+1x1ee
        """
        out = []
        for mul, ir in self:
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            elif mul > 0:
                out.append((mul, ir))
        return Irreps(out)

    def remove_zero_multiplicities(self):
        """Remove any irreps with multiplicities of zero.

        Returns
        -------
        `e3nn.o3.Irreps`

        Examples
        --------

        >>> Irreps("4x0ee + 0x1oe + 2x3ee").remove_zero_multiplicities()
        4x0ee+2x3ee

        """
        out = [(mul, ir) for mul, ir in self if mul > 0]
        return Irreps(out)

    def sort(self):
        r"""Sort the representations.

        Returns
        -------
        irreps : `e3nn.o3.Irreps`
        p : tuple of int; index og -> sort
        inv : tuple of int; index sort -> og

        Examples
        --------

        >>> Irreps("1ee + 0ee + 1ee").sort().irreps
        1x0ee+1x1ee+1x1ee

        >>> Irreps("2oe + 1ee + 0ee + 1ee").sort().p
        (3, 1, 0, 2)

        >>> Irreps("2o + 1e + 0e + 1e").sort().inv
        (2, 1, 3, 0)
        """
        Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
        out = [(ir, i, mul) for i, (mul, ir) in enumerate(self)]
        out = sorted(out)
        inv = tuple(i for _, i, _ in out)
        p = perm.inverse(inv)
        irreps = Irreps([(mul, ir) for ir, _, mul in out])
        return Ret(irreps, p, inv)

    @property
    def dim(self) -> int:
        return sum(mul * ir.dim for mul, ir in self)

    @property
    def num_irreps(self) -> int:
        return sum(mul for mul, _ in self)

    @property
    def ls(self) -> List[int]:
        # Return the l list
        return [l for mul, (l, p, t) in self for _ in range(mul)]

    @property
    def lmax(self) -> int:
        if len(self) == 0:
            raise ValueError("Cannot get lmax of empty Irreps")
        return max(self.ls)

    def __repr__(self):
        return "+".join(f"{mul_ir}" for mul_ir in self)

    def D_from_angles(self, alpha, beta, gamma, k=None, kt=None):
        r"""Matrix of the representation

        Parameters
        ----------
        alpha : `torch.Tensor`
            tensor of shape :math:`(...)`

        beta : `torch.Tensor`
            tensor of shape :math:`(...)`

        gamma : `torch.Tensor`
            tensor of shape :math:`(...)`

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        kt : `torch.Tensor`, optional
            tensor of shape :math:`(...)`
            How many times the time reversal operation is applied.

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return direct_sum(*[ir.D_from_angles(alpha, beta, gamma, k, kt) for mul, ir in self for _ in range(mul)])

    def D_from_quaternion(self, q, k=None, kt=None):
        r"""Matrix of the representation

        Parameters
        ----------
        q : `torch.Tensor`
            tensor of shape :math:`(..., 4)`

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        kt : `torch.Tensor`, optional
            tensor of shape :math:`(...)`
            How many times the time reversal operation is applied.

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return self.D_from_angles(*_rotation.quaternion_to_angles(q), k, kt)

    def D_from_matrix(self, R, parity=True, time_reversal=False):
        r"""Matrix of the representation.

        Determination of the Matrix: (k in the func) 1: do reverse; 0: remain the same

        parity and time_reversal to control which reverse is about

        Parameters
        ----------
        R : `torch.Tensor`
            tensor of shape :math:`(..., 3, 3)`

        parity: `bool`
            matrix consider parity

        time_reversal: `bool`
            matrix consider time_reversal

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        d = torch.det(R).sign()
        R = d[..., None, None] * R
        k = (1 - d) / 2
        kp = k if parity else torch.as_tensor(0).type_as(k)
        kt = k if time_reversal else torch.as_tensor(0).type_as(k)
        return self.D_from_angles(*_rotation.matrix_to_angles(R), kp, kt)

    def D_from_axis_angle(self, axis, angle):
        r"""Matrix of the representation

        Parameters
        ----------
        axis : `torch.Tensor`
            tensor of shape :math:`(..., 3)`

        angle : `torch.Tensor`
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return self.D_from_angles(*_rotation.axis_angle_to_angles(axis, angle))

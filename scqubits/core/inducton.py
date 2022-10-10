# inducton.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import math
import os

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy as sp

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray

import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plot_defaults as defaults
import scqubits.utils.plotting as plot

from scqubits.core.discretization import Grid1d
from scqubits.core.noise import NoisySystem
from scqubits.core.storage import WaveFunction

LevelsTuple = Tuple[int, ...]
Transition = Tuple[int, int]
TransitionsTuple = Tuple[Transition, ...]

# —quantum phase slip box / inducton——————————————————————————————————————————————


class Inducton(base.QubitBaseClass1d, serializers.Serializable, NoisySystem):
    r"""Class for the QPS Box and inducton qubit. The Hamiltonian is
    represented in dense form in the fluxoid basis,
    :math:`H_\text{QPSB}=\frac{E_\text{L}}{2}(2\pi)^2(\hat{m}-f)^2-\frac{E_\text{S}}{2}(
    |m\rangle\langle m+1|+\text{h.c.})`
    Initialize with, for example::

        Inducton(ES=1.0, EC=0.4, f=0.2, mcut=30)

    Parameters
    ----------
    ES:
       phase slip energy
    EL:
        inductive energy
    f:
        applied flux
    mcut:
        fluxoid basis cutoff, `m = -mcut, ..., mcut`
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
    """
    ES = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EL = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    f = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    mcut = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        ES: float,
        EL: float,
        f: float,
        mcut: int,
        truncated_dim: int = 6,
        id_str: Optional[str] = None,
    ) -> None:
        base.QuantumSystem.__init__(self, id_str=id_str)
        self.ES = ES
        self.EL = EL
        self.f = f
        self.mcut = mcut
        self.truncated_dim = truncated_dim
        self._default_grid = discretization.Grid1d(-np.pi, np.pi, 151)
        self._default_m_range = (-5, 6)
        self._image_filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "qubit_img/fixed-inducton.jpg"
        )

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {"ES": 15.0, "EL": 0.3, "f": 0.0, "mcut": 30, "truncated_dim": 10}

    @classmethod
    def supported_noise_channels(cls) -> List[str]:
        """Return a list of supported noise channels"""
        return [
            "tphi_1_over_f_cc",
            "tphi_1_over_f_ng",
            "t1_capacitive",
            "t1_charge_impedance",
        ]

    @classmethod
    def effective_noise_channels(cls) -> List[str]:
        """Return a default list of channels used when calculating effective t1 and
        t2 noise."""
        noise_channels = cls.supported_noise_channels()
        noise_channels.remove("t1_charge_impedance")
        return noise_channels

    def _hamiltonian_diagonal(self) -> ndarray:
        dimension = self.hilbertdim()
        return (2*np.pi) ** 2 * 0.5 * self.EL * (np.arange(dimension) - self.mcut - self.f) ** 2

    def _hamiltonian_offdiagonal(self) -> ndarray:
        dimension = self.hilbertdim()
        return np.full(shape=(dimension - 1,), fill_value=-self.ES / 2.0)

    def _evals_calc(self, evals_count: int) -> ndarray:
        diagonal = self._hamiltonian_diagonal()
        off_diagonal = self._hamiltonian_offdiagonal()

        evals = sp.linalg.eigvalsh_tridiagonal(
            diagonal,
            off_diagonal,
            select="i",
            select_range=(0, evals_count - 1),
            check_finite=False,
        )
        return evals

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:
        diagonal = self._hamiltonian_diagonal()
        off_diagonal = self._hamiltonian_offdiagonal()

        evals, evecs = sp.linalg.eigh_tridiagonal(
            diagonal,
            off_diagonal,
            select="i",
            select_range=(0, evals_count - 1),
            check_finite=False,
        )
        return evals, evecs

    def m_operator(self) -> ndarray:
        """Returns charge operator `m` in the charge basis"""
        diag_elements = np.arange(-self.mcut, self.mcut + 1, 1)
        return np.diag(diag_elements)

    def exp_i_n_operator(self) -> ndarray:
        """Returns operator :math:`e^{i\\varn}` in the charge basis"""
        dimension = self.hilbertdim()
        entries = np.repeat(1.0, dimension - 1)
        exp_op = np.diag(entries, -1)
        return exp_op

    def cos_n_operator(self) -> ndarray:
        """Returns operator :math:`\\cos \\varn` in the charge basis"""
        cos_op = 0.5 * self.exp_i_n_operator()
        cos_op += cos_op.T
        return cos_op

    def sin_n_operator(self) -> ndarray:
        """Returns operator :math:`\\sin \\varn` in the charge basis"""
        sin_op = -1j * 0.5 * self.exp_i_n_operator()
        sin_op += sin_op.conjugate().T
        return sin_op

    def hamiltonian(self) -> ndarray:
        """Returns Hamiltonian in fluxoid basis"""
        dimension = self.hilbertdim()
        hamiltonian_mat = np.diag(
            [
                (2*np.pi) ** 2 * 0.5 * self.EL * (ind - self.mcut - self.f) ** 2
                for ind in range(dimension)
            ]
        )
        ind = np.arange(dimension - 1)
        hamiltonian_mat[ind, ind + 1] = -self.ES / 2.0
        hamiltonian_mat[ind + 1, ind] = -self.ES / 2.0
        return hamiltonian_mat

    def d_hamiltonian_d_ng(self) -> ndarray:
        """Returns operator representing a derivative of the Hamiltonian with respect to
        flux offset `f`."""
        return - (2*np.pi) ** 2 * self.EL * self.m_operator()

    def d_hamiltonian_d_EJ(self) -> ndarray:
        """Returns operator representing a derivative of the Hamiltonian with respect
        to ES."""
        return -self.cos_phi_operator()

    def hilbertdim(self) -> int:
        """Returns Hilbert space dimension"""
        return 2 * self.mcut + 1

    def potential(self, n: Union[float, ndarray]) -> ndarray:
        """Inducton charge-basis potential evaluated at `n`.

        Parameters
        ----------
        n:
            charge variable value as conjugate to m
        """
        return -self.ES * np.cos(n)

    def plot_m_wavefunction(
        self,
        esys: Tuple[ndarray, ndarray] = None,
        mode: str = "real",
        which: int = 0,
        mrange: Tuple[int, int] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Plots inudcton wave function in fluxoid basis

        Parameters
        ----------
        esys:
            eigenvalues, eigenvectors
        mode:
            `'abs_sqr', 'abs', 'real', 'imag'`
        which:
             index or indices of wave functions to plot (default value = 0)
        nrange:
             range of `m` to be included on the x-axis (default value = (-5,6))
        **kwargs:
            plotting parameters
        """
        if mrange is None:
            mrange = self._default_m_range
        m_wavefunc = self.numberbasis_wavefunction(esys, which=which)
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        m_wavefunc.amplitudes = amplitude_modifier(m_wavefunc.amplitudes)
        kwargs = {
            **defaults.wavefunction1d_discrete(mode, basis="m"),
            **kwargs,
        }  # if any duplicates, later ones survive
        return plot.wavefunction1d_discrete(m_wavefunc, xlim=mrange, **kwargs)

    def plot_n_wavefunction(
        self,
        esys: Tuple[ndarray, ndarray] = None,
        which: int = 0,
        n_grid: Grid1d = None,
        mode: str = "abs_sqr",
        scaling: float = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Alias for plot_wavefunction"""
        return self.plot_wavefunction_n(
            esys=esys,
            which=which,
            n_grid=n_grid,
            mode=mode,
            scaling=scaling,
            **kwargs
        )

    def numberbasis_wavefunction(
        self, esys: Tuple[ndarray, ndarray] = None, which: int = 0
    ) -> WaveFunction:
        """Return the inducton wave function in number basis. The specific index of the
        wave function to be returned is `which`.

        Parameters
        ----------
        esys:
            if `None`, the eigensystem is calculated on the fly; otherwise, the provided
            eigenvalue, eigenvector arrays as obtained from `.eigensystem()`,
            are used (default value = None)
        which:
            eigenfunction index (default value = 0)
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            esys = self.eigensys(evals_count=evals_count)
        evals, evecs = esys

        m_vals = np.arange(-self.mcut, self.mcut + 1)
        return storage.WaveFunction(m_vals, evecs[:, which], evals[which])

    def wavefunction(
        self,
        esys: Optional[Tuple[ndarray, ndarray]] = None,
        which: int = 0,
        n_grid: Grid1d = None,
    ) -> WaveFunction:
        """Return the inducton wave function in charge basis. The specific index of the
        wavefunction is `which`. `esys` can be provided, but if set to `None` then it is
        calculated on the fly.

        Parameters
        ----------
        esys:
            if None, the eigensystem is calculated on the fly; otherwise, the provided
            eigenvalue, eigenvector arrays as obtained from `.eigensystem()` are used
        which:
            eigenfunction index (default value = 0)
        n_grid:
            used for setting a custom grid for phi; if None use self._default_grid
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            evals, evecs = self.eigensys(evals_count=evals_count)
        else:
            evals, evecs = esys

        n_wavefunc = self.numberbasis_wavefunction(esys, which=which)

        n_grid = n_grid or self._default_grid
        n_basis_labels = n_grid.make_linspace()
        n_wavefunc_amplitudes = np.empty(n_grid.pt_count, dtype=np.complex_)
        for k in range(n_grid.pt_count):
            n_wavefunc_amplitudes[k] = (1j**which / math.sqrt(2 * np.pi)) * np.sum(
                n_wavefunc.amplitudes
                * np.exp(1j * n_basis_labels[k] * n_wavefunc.basis_labels)
            )
        return storage.WaveFunction(
            basis_labels=n_basis_labels,
            amplitudes=n_wavefunc_amplitudes,
            energy=evals[which],
        )

    def _compute_dispersion(
        self,
        dispersion_name: str,
        param_name: str,
        param_vals: ndarray,
        transitions_tuple: TransitionsTuple = ((0, 1),),
        levels_tuple: Optional[LevelsTuple] = None,
        point_count: int = 50,
        num_cpus: Optional[int] = None,
    ) -> Tuple[ndarray, ndarray]:
        if dispersion_name != "ng":
            return super()._compute_dispersion(
                dispersion_name,
                param_name,
                param_vals,
                transitions_tuple=transitions_tuple,
                levels_tuple=levels_tuple,
                point_count=point_count,
                num_cpus=num_cpus,
            )

        max_level = (
            np.max(transitions_tuple) if levels_tuple is None else np.max(levels_tuple)
        )
        previous_ng = self.ng
        self.ng = 0.0
        specdata_ng_0 = self.get_spectrum_vs_paramvals(
            param_name,
            param_vals,
            evals_count=max_level + 1,
            get_eigenstates=False,
            num_cpus=num_cpus,
        )
        self.ng = 0.5
        specdata_ng_05 = self.get_spectrum_vs_paramvals(
            param_name,
            param_vals,
            evals_count=max_level + 1,
            get_eigenstates=False,
            num_cpus=num_cpus,
        )
        self.ng = previous_ng

        if levels_tuple is not None:
            dispersion = np.asarray(
                [
                    [
                        np.abs(
                            specdata_ng_0.energy_table[param_index, j]
                            - specdata_ng_05.energy_table[param_index, j]
                        )
                        for param_index, _ in enumerate(param_vals)
                    ]
                    for j in levels_tuple
                ]
            )
            return specdata_ng_0.energy_table, dispersion

        dispersion_list = []
        for i, j in transitions_tuple:
            list_ij = []
            for param_index, _ in enumerate(param_vals):
                ei_0 = specdata_ng_0.energy_table[param_index, i]
                ei_05 = specdata_ng_05.energy_table[param_index, i]
                ej_0 = specdata_ng_0.energy_table[param_index, j]
                ej_05 = specdata_ng_05.energy_table[param_index, j]
                list_ij.append(
                    np.max([np.abs(ei_0 - ej_0), np.abs(ei_05 - ej_05)])
                    - np.min([np.abs(ei_0 - ej_0), np.abs(ei_05 - ej_05)])
                )
            dispersion_list.append(list_ij)
        return specdata_ng_0.energy_table, np.asarray(dispersion_list)


# — Charge-tunable inducton ———————————————————————————————————————————


class TunableInducton(Inducton, serializers.Serializable, NoisySystem):
    r"""Class for the charge-tunable inducton qubit. The Hamiltonian is represented in
    dense form in the number basis, :math:`H_\text{QPSB}=\frac{E_\text{L}}{2}(2\pi)^2(\hat{m}-f)^2-
    \frac{E_\text{S}(V_g)}{2}(|m\rangle\langle m+1|+\text{h.c.})` ,
    Here, the effective quantum phase slip energy is charge-tunable: :math:`\mathcal{
    E}_S(V_g) = E_{S,\text{max}} \sqrt{\cos^2(\pi Q_g/2e) + d^2 \sin^2(
    \pi Q_g/2e)}` where :math: `Q_g = C_gV_g` defines the charge tunable bias
    and :math:`d=(E_{S2}-E_{S1})(E_{S1}+E_{S2})` parametrizes the phase slip junction asymmetry.

    Initialize with, for example::

        TunableInducton(ESmax=1.0, d=0.1, EL=2.0, Q=0.3, f=0.2, mcut=30)

    Parameters
    ----------
    ESmax:
       maximum effective quantum phase slip energy (sum of the quantum phase slip energies of the two
       junctions)
    d:
        junction asymmetry parameter
    EL:
        inductive energy
    Q:
        gate-voltage induced charge bias, in units of the cooper pair charge
    f:
        offset charge
    mcut:
        fluxoid basis cutoff, `m = -mcut, ..., mcut`
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
    """
    ESmax = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    d = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    Q = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        ESmax: float,
        EL: float,
        d: float,
        Q: float,
        f: float,
        mcut: int,
        truncated_dim: int = 6,
        id_str: Optional[str] = None,
    ) -> None:
        base.QuantumSystem.__init__(self, id_str=id_str)
        self.ESmax = ESmax
        self.EL = EL
        self.d = d
        self.Q = Q
        self.f = f
        self.mcut = mcut
        self.truncated_dim = truncated_dim
        self._default_grid = discretization.Grid1d(-np.pi, np.pi, 151)
        self._default_m_range = (-5, 6)
        self._image_filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "qubit_img/tunable-inducton.jpg"
        )

    @property
    def ES(self) -> float:  # type: ignore
        """This is the effective, charge dependent quantum phase slip energy, playing the role
        of ES in the parent class `Inducton`"""
        return self.ESmax * np.sqrt(
            np.cos(np.pi * self.Q) ** 2
            + self.d**2 * np.sin(np.pi * self.Q) ** 2
        )

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "ESmax": 20.0,
            "EL": 0.3,
            "d": 0.01,
            "Q": 0.0,
            "f": 0.0,
            "mcut": 30,
            "truncated_dim": 10,
        }

    @classmethod
    def supported_noise_channels(cls) -> List[str]:
        """Return a list of supported noise channels"""
        return [
            "tphi_1_over_f_flux",
            "tphi_1_over_f_cc",
            "tphi_1_over_f_ng",
            "t1_capacitive",
            "t1_flux_bias_line",
            "t1_charge_impedance",
        ]

    def d_hamiltonian_d_Q(self) -> ndarray:
        """Returns operator representing a derivative of the Hamiltonian with respect
        to `charge`."""
        return (
            np.pi
            * self.ESmax
            * np.cos(np.pi * self.Q)
            * np.sin(np.pi * self.Q)
            * (self.d**2 - 1)
            / np.sqrt(
                np.cos(np.pi * self.Q) ** 2
                + self.d**2 * np.sin(np.pi * self.Q) ** 2
            )
            * self.cos_n_operator()
        )

    def _compute_dispersion(
        self,
        dispersion_name: str,
        param_name: str,
        param_vals: ndarray,
        transitions_tuple: TransitionsTuple = ((0, 1),),
        levels_tuple: Optional[LevelsTuple] = None,
        point_count: int = 50,
        num_cpus: Optional[int] = None,
    ) -> Tuple[ndarray, ndarray]:
        if dispersion_name != "flux":
            return super()._compute_dispersion(
                dispersion_name,
                param_name,
                param_vals,
                transitions_tuple=transitions_tuple,
                levels_tuple=levels_tuple,
                point_count=point_count,
                num_cpus=num_cpus,
            )

        max_level = (
            np.max(transitions_tuple) if levels_tuple is None else np.max(levels_tuple)
        )
        previous_flux = self.flux
        self.flux = 0.0
        specdata_flux_0 = self.get_spectrum_vs_paramvals(
            param_name,
            param_vals,
            evals_count=max_level + 1,
            get_eigenstates=False,
            num_cpus=num_cpus,
        )
        self.flux = 0.5
        specdata_flux_05 = self.get_spectrum_vs_paramvals(
            param_name,
            param_vals,
            evals_count=max_level + 1,
            get_eigenstates=False,
            num_cpus=num_cpus,
        )
        self.flux = previous_flux

        if levels_tuple is not None:
            dispersion = np.asarray(
                [
                    [
                        np.abs(
                            specdata_flux_0.energy_table[param_index, j]  # type:ignore
                            - specdata_flux_05.energy_table[
                                param_index, j
                            ]  # type:ignore
                        )
                        for param_index, _ in enumerate(param_vals)
                    ]
                    for j in levels_tuple
                ]
            )
            return specdata_flux_0.energy_table, dispersion  # type:ignore

        dispersion_list = []
        for i, j in transitions_tuple:
            list_ij = []
            for param_index, _ in enumerate(param_vals):
                ei_0 = specdata_flux_0.energy_table[param_index, i]  # type:ignore
                ei_05 = specdata_flux_05.energy_table[param_index, i]  # type:ignore
                ej_0 = specdata_flux_0.energy_table[param_index, j]  # type:ignore
                ej_05 = specdata_flux_05.energy_table[param_index, j]  # type:ignore
                list_ij.append(
                    np.max([np.abs(ei_0 - ej_0), np.abs(ei_05 - ej_05)])
                    - np.min([np.abs(ei_0 - ej_0), np.abs(ei_05 - ej_05)])
                )
            dispersion_list.append(list_ij)
        return specdata_flux_0.energy_table, np.asarray(dispersion_list)  # type:ignore

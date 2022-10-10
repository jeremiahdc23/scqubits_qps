# plasmonium.py
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

import cmath
import math
import os

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy as sp

from numpy import ndarray

import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.operators as op
import scqubits.core.oscillator as osc
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers

from scqubits.core.noise import NoisySystem

if TYPE_CHECKING:
    from scqubits.core.discretization import Grid1d


class Plasmonium(base.QubitBaseClass1d, serializers.Serializable, NoisySystem):
    r"""Class for the plasmonium qubit. Hamiltonian :math:`H_\text{pl}=4E_\text{
    C}n^2-E_\text{S}\cos(2\pi(n - n_g)) - \frac{E_L}{2}\partial_n^2`
    is represented in dense form. The employed basis is the EC-EL harmonic oscillator
    basis. The cosine term in the potential is handled via matrix exponentiation.
    Initialize with, for example::

        qubit = Plasmonium(ES=1.0, EC=2.0, EL=0.3, ng=0.2, cutoff=120)

    Parameters
    ----------
    ES: float
        quantum phase slip energy
    EC: float
        charging energy
    EL: float
        inductive energy
    ng: float
        gate-voltage induced charge in units of one cooper pair charge
    cutoff: int
        number of harm. osc. basis states used in diagonalization
    truncated_dim: int
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str: str
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
    """
    ES = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EC = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    EL = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ng = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    cutoff = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")

    def __init__(
        self,
        ES: float,
        EC: float,
        EL: float,
        ng: float,
        cutoff: int,
        truncated_dim: int = 6,
        id_str: Optional[str] = None,
    ) -> None:
        base.QuantumSystem.__init__(self, id_str=id_str)
        self.ES = ES
        self.EC = EC
        self.EL = EL
        self.ng = ng
        self.cutoff = cutoff
        self.truncated_dim = truncated_dim
        self._default_grid = discretization.Grid1d(-4.5 * np.pi, 4.5 * np.pi, 151)
        self._image_filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "qubit_img/plasmonium.jpg"
        )

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "ES": 8.9,
            "EC": 2.5,
            "EL": 0.5,
            "ng": 0.0,
            "cutoff": 110,
            "truncated_dim": 10,
        }

    @classmethod
    def supported_noise_channels(cls) -> List[str]:
        """Return a list of supported noise channels"""
        return [
            "tphi_1_over_f_cc",
            "tphi_1_over_f_flux",
            "t1_capacitive",
            "t1_charge_impedance",
            "t1_flux_bias_line",
            "t1_inductive",
            "t1_quasiparticle_tunneling",
        ]

    @classmethod
    def effective_noise_channels(cls) -> List[str]:
        """Return a default list of channels used when calculating effective t1 and t2
        noise."""
        noise_channels = cls.supported_noise_channels()
        noise_channels.remove("t1_charge_impedance")
        return noise_channels

    def phi_osc(self) -> float:
        """
        Returns
        -------
            Returns oscillator length for the LC oscillator composed of the plasmonium
             inductance and capacitance.
        """
        return (8.0 * self.EC / self.EL) ** 0.25  # LC oscillator length

    def plasma_energy(self) -> float:
        """
        Returns
        -------
            Returns the plasma oscillation frequency, sqrt(8*EL*EC).
        """
        return math.sqrt(8.0 * self.EL * self.EC)  # LC plasma oscillation energy

    def phi_operator(self) -> ndarray:
        """
        Returns
        -------
            Returns the phi operator in the LC harmonic oscillator basis
        """
        dimension = self.hilbertdim()
        return (
                (op.creation(dimension) + op.annihilation(dimension))
                * self.phi_osc()
                / math.sqrt(2)
        )

    def n_operator(self) -> ndarray:
        """
        Returns
        -------
            Returns the :math:`n = - i d/d\\phi` operator in the LC harmonic
            oscillator basis
        """
        dimension = self.hilbertdim()
        return (
                1j
                * (op.creation(dimension) - op.annihilation(dimension))
                / (self.phi_osc() * math.sqrt(2))
        )

    def exp_i_n_operator(self, alpha: float = 1.0, beta: float = 0.0) -> ndarray:
        """
        Returns
        -------
            Returns the :math:`e^{i (\\alpha n + \\beta) }` operator in the
            LC harmonic oscillator basis,
            with :math:`\\alpha` and :math:`\\beta` being numbers
        """
        exponent = 1j * (alpha * self.n_operator())
        return sp.linalg.expm(exponent) * cmath.exp(1j * beta)

    def cos_n_operator(self, alpha: float = 1.0, beta: float = 0.0) -> ndarray:
        """
        Returns
        -------
            Returns the :math:`\\cos (\\alpha n + \\beta)` operator in the LC
            harmonic oscillator basis,
            with :math:`\\alpha` and :math:`\\beta` being numbers
        """
        argument = alpha * self.n_operator() + beta * np.eye(self.hilbertdim())
        return sp.linalg.cosm(argument)

    def sin_n_operator(self, alpha: float = 1.0, beta: float = 0.0) -> ndarray:
        """
        Returns
        -------
            Returns the :math:`\\sin (\\alpha n + \\beta)` operator in the
            LC harmonic oscillator basis
            with :math:`\\alpha` and :math:`\\beta` being numbers
        """
        argument = alpha * self.n_operator() + beta * np.eye(self.hilbertdim())
        return sp.linalg.sinm(argument)

    def hamiltonian(self) -> ndarray:  # follow Zhu et al., PRB 87, 024510 (2013)
        """Construct Hamiltonian matrix in harmonic-oscillator basis, following Zhu
        et al., PRB 87, 024510 (2013)"""
        dimension = self.hilbertdim()
        diag_elements = [(i + 0.5) * self.plasma_energy() for i in range(dimension)]
        lc_osc_matrix = np.diag(diag_elements)

        cos_matrix = self.cos_n_operator(beta=2 * np.pi * self.ng)

        hamiltonian_mat = lc_osc_matrix - self.ES * cos_matrix
        return hamiltonian_mat

    def d_hamiltonian_d_ES(self) -> ndarray:
        """Returns operator representing a derivative of the Hamiltonian with respect
        to `ES`.

        The charge is grouped as in the Hamiltonian.
        """
        return -self.cos_phi_operator(1, 2 * np.pi * self.ng)

    def d_hamiltonian_d_ng(self) -> ndarray:
        """Returns operator representing a derivative of the Hamiltonian with respect
        to `ng`.

        The charge is grouped as in the Hamiltonian.
        """
        return -2 * np.pi * self.ES * self.sin_n_operator(1, 2 * np.pi * self.ng)

    def hilbertdim(self) -> int:
        """
        Returns
        -------
            Returns the Hilbert space dimension."""
        return self.cutoff

    def potential(self, n: Union[float, ndarray]) -> ndarray:
        """Plasmonium potential evaluated at `n`.

        Parameters
        ----------
            float value of the phase variable `n`

        Returns
        -------
        float or ndarray
        """
        return 4 * self.EC * n * n - self.ES * np.cos(
            2.0 * np.pi (n + self.ng)
        )

    def wavefunction(
        self,
        esys: Optional[Tuple[ndarray, ndarray]],
        which: int = 0,
        n_grid: "Grid1d" = None,
    ) -> storage.WaveFunction:
        """Returns a plasmonium wave function in `n` basis

        Parameters
        ----------
        esys:
            eigenvalues, eigenvectors
        which:
             index of desired wave function (default value = 0)
        n_grid:
            used for setting a custom grid for n; if None use self._default_grid
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            evals, evecs = self.eigensys(evals_count=evals_count)
        else:
            evals, evecs = esys
        dim = self.hilbertdim()

        n_grid = n_grid or self._default_grid

        n_basis_labels = n_grid.make_linspace()
        wavefunc_osc_basis_amplitudes = evecs[:, which]
        n_wavefunc_amplitudes = np.zeros(n_grid.pt_count, dtype=np.complex_)
        phi_osc = self.phi_osc()
        for n in range(dim):
            n_wavefunc_amplitudes += wavefunc_osc_basis_amplitudes[
                n
            ] * osc.harm_osc_wavefunction(n, n_basis_labels, phi_osc)
        return storage.WaveFunction(
            basis_labels=n_basis_labels,
            amplitudes=n_wavefunc_amplitudes,
            energy=evals[which],
        )

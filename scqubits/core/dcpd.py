# dcp.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import math
import os

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm, eigsh
from scipy.special import kn
import matplotlib.pyplot as plt

import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.harmonic_osc as osc
import scqubits.core.operators as op
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plotting as plot
import scqubits.utils.spectrum_utils as spec_utils
import scqubits.utils.plot_defaults as defaults
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scqubits.utils.spectrum_utils import matrix_element


# —Double Cooper pair tunneling qubit ————————————————————————
class Dcpd(base.QubitBaseClass, serializers.Serializable):
    r"""double Cooper pair tunneling qubit

    | [1] Smith et al., NPJ Quantum Inf. 6, 8 (2020) http://www.nature.com/articles/s41534-019-0231-2

    .. math::

        H H_\text{dcp} = 4E_\text{C}[2n_\phi^2+\frac{1}{2}(n_\varphi-N_\text{g}-n_\theta)^2+xn_\theta^2]
                           +E_\text{L}(\frac{1}{4}\phi^2+\theta^2)
                           -2E_\text{J}\cos(\varphi)\cos(\frac{\phi}{2}+\frac{\varphi_\text{ext}}{2})

    The employed basis are harmonic basis for :math:`\phi,\theta` and charge basis for :math:`\varphi`. The cosine term in the
    potential is handled via matrix exponentiation. Initialize with, for example::

        qubit = Dcp(EJ=15.0, EC=2.0, EL=1.0, x=0.02, dC=0, dL=0, dJ=0, flux=0.5, Ng=0, N0=7, q0=30, p0=7)

    Parameters
    ----------
    EJ: float
        Josephson energy
    EC: float
        charging energy
    EL: float
        inductive energy
    ELA: float
        additional inductive energy
    x: float
        ratio of the junction capacitance to the shunt capacitance x = C_J / C_shunt
    dC: float
        disorder in capacitance, i.e., EC / (1 \pm dC)
    dL: float
        disorder in inductance, i.e., EL / (1 \pm dL)
    dJ: float
        disorder in junction energy, i.e., EJ * (1 \pm dJ)
    flux: float
        external magnetic flux in angular units, 2pi corresponds to one flux quantum
    Ng: float
        offset charge
    N0: int
        number of charge states used in diagonalization, -N0 <= n_\varphi <= N0
    q0: int
        number of harmonic oscillator basis used in diagonalization of \theta
    p0: int
        number of harmonic oscillator basis used in diagonalization of \phi
    truncated_dim: int, optional
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    """
    EJ = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EC = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    EL = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    ELA = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    x = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    flux = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    N0 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    q0 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')
    p0 = descriptors.WatchedProperty('QUANTUMSYSTEM_UPDATE')

    def __init__(self, EJ, EC, EL, ELA, x, dL, dC, dJ, flux, fluxa, kbt, truncated_dim=None):
        self.EJ = EJ
        self.EC = EC
        self.EL = EL
        self.ELA = ELA
        self.x = x
        self.dL = dL
        self.dC = dC
        self.dJ = dJ
        self.flux = flux
        self.fluxa = fluxa
        self.kbt = kbt * 1e-3 * 1.38e-23 / 6.63e-34 / 1e9  # temperature unit mK
        self.phi_grid = discretization.Grid1d(-8 * np.pi, 8 * np.pi, 100)
        self.varphi_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        self.theta_grid = discretization.Grid1d(-4 * np.pi, 4 * np.pi, 100)
        self.theta_cut = 3
        self.ph = 0
        self.truncated_dim = truncated_dim
        self._sys_type = type(self).__name__
        self._evec_dtype = np.float_
        self._image_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'qubit_pngs/double_cooper_pair_tunneling_qubit.png')

    @staticmethod
    def default_params():
        return {
            'EJ': 4.5,
            'EC': 1.05,
            'EL': 0.1,
            'ELA': 0.1,
            'x': 10,
            'dL': 0,
            'dC': 0,
            'dJ': 0,
            'flux': 0,
            'fluxa': 0.5,
            'theta_cut': 3,
            'truncated_dim': 10
        }

    @staticmethod
    def nonfit_params():
        return ['flux', 'truncated_dim']

    def dim_phi(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension of :math:`phi' degree of freedom."""
        return self.phi_grid.pt_count

    def dim_theta(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension of :math:`theta' degree of freedom."""
        return self.theta_cut

    def dim_varphi(self):
        """
        Returns
        -------
        int
            Returns the Hilbert space dimension of :math:`varphi' degree of freedom."""
        return self.varphi_grid.pt_count

    def hilbertdim(self):
        """
        Returns
        -------
        int
            Returns the total Hilbert space dimension."""
        return self.dim_phi() * self.dim_theta() * self.dim_varphi()

    def _dis_el(self):
        """
        Returns
        -------
        float
            Returns the inductive energy renormalized by with disorder."""
        return self.EL / (1 - self.dL ** 2)

    def _dis_ec(self):
        """
        Returns
        -------
        float
            Returns the capacitance energy renormalized by with disorder."""
        return self.EC / (1 - self.dC ** 2)

    def phi_osc(self):
        """
        Returns
        -------
        float
            Returns the oscillator strength of :math:`phi' degree of freedom."""
        return (32 * self._dis_ec() / self._dis_el()) ** 0.25

    def theta_osc(self):
        """
        Returns
        -------
        float
            Returns the oscillator strength of :math:`theta' degree of freedom."""
        return (8 * self._dis_ec() * self.x / (2 * self._dis_el() + self.ELA)) ** 0.25

    def varphi_osc(self):
        """Return the oscillator strength of varphi degree of freedom"""
        return (4 * self._dis_ec() / self.ELA) ** 0.25

    def phi_plasma(self):
        """
        Returns
        -------
        float
            Returns the plasma oscillation frequency of :math:`phi' degree of freedom.
        """
        return math.sqrt(8.0 * self._dis_el() * self._dis_ec())

    def theta_plasma(self):
        """
        Returns
        -------
        float
            Returns the plasma oscillation frequency of :math:`theta' degree of freedom.
        """
        return math.sqrt(8.0 * self.x * self._dis_ec() * (2 * self._dis_el() + self.ELA))

    def varphi_plasma(self):
        """
        Returns
        -------
        float
            Returns the plasma oscillation frequency for the varphi degree of freedom.
        """
        return math.sqrt(4.0 * self.ELA * self._dis_ec())

    def _phi_operator(self):
        return sparse.dia_matrix((self.phi_grid.make_linspace(), [0]), shape=(self.dim_phi(), self.dim_phi())).tocsc()

    def phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`phi' operator in total Hilbert space
        """
        return self._kron3(self._phi_operator(), self._identity_theta(), self._identity_varphi())

    def _n_phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_\phi = - i d/d\\phi` operator
        """
        return self.phi_grid.first_derivative_matrix(prefactor=-1j)

    def n_phi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_phi' operator in total Hilbert space
        """
        return self._kron3(self._n_phi_operator(), self._identity_theta(), self._identity_varphi())

    def _n_phi_2_operator(self):
        return self.phi_grid.second_derivative_matrix(prefactor=- 1)

    def n_phi_2_operator(self):
        return self._kron3(self._n_phi_2_operator(), self._identity_theta(), self._identity_varphi())

    def _n_varphi_2_operator(self):
        return self.varphi_grid.second_derivative_matrix(prefactor=- 1)

    def n_varphi_2_operator(self):
        return self._kron3(self._identity_phi(), self._identity_theta(), self._n_varphi_2_operator())

    def _theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`theta' operator in the LC harmonic oscillator basis
        """
        dimension = self.dim_theta()
        return (op.creation_sparse(dimension) + op.annihilation_sparse(dimension)) * self.theta_osc() / math.sqrt(2)

    def theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`theta' operator in total Hilbert space
        """
        return self._kron3(self._identity_phi(), self._theta_operator(), self._identity_varphi())

    def _n_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_\theta = - i d/d\\theta` operator in the LC harmonic oscillator basis
        """
        dimension = self.dim_theta()
        return 1j * (op.creation_sparse(dimension) - op.annihilation_sparse(dimension)) / (
                self.theta_osc() * math.sqrt(2))

    def n_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_theta' operator in total Hilbert space
        """
        return self._kron3(self._identity_phi(), self._n_theta_operator(), self._identity_varphi())

    def n_theta_2_operator(self):
        return self.n_theta_operator() ** 2

    def _cos_phi_div_operator(self, div):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\cos \\phi/div` operator
        """
        cos_phi_div_vals = np.cos(self.phi_grid.make_linspace() / div)
        return sparse.dia_matrix((cos_phi_div_vals, [0]), shape=(self.dim_phi(), self.dim_phi())).tocsc()

    def _sin_phi_div_operator(self, div):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\sin \\phi/div` operator
        """
        sin_phi_div_vals = np.sin(self.phi_grid.make_linspace() / div)
        return sparse.dia_matrix((sin_phi_div_vals, [0]), shape=(self.dim_phi(), self.dim_phi())).tocsc()

    def _exp_i_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`e^{i\\theta}` operator in the LC harmonic oscillator basis
        """
        exponent = 1j * self._theta_operator()
        return expm(exponent)

    def _cos_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\cos \\theta` operator in the LC harmonic oscillator basis
        """
        cos_theta_op = 0.5 * self._exp_i_theta_operator()
        cos_theta_op += cos_theta_op.conj().T
        return np.real(cos_theta_op)

    def _sin_theta_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\sin \\theta` operator in the LC harmonic oscillator basis
        """
        sin_theta_op = -1j * 0.5 * self._exp_i_theta_operator()
        sin_theta_op += sin_theta_op.conj().T
        return np.real(sin_theta_op)

    def _varphi_operator(self):
        return sparse.dia_matrix((self.varphi_grid.make_linspace(), [0]),
                                 shape=(self.dim_varphi(), self.dim_varphi())).tocsc()

    def varphi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`varphi' operator in total Hilbert space
        """
        return self._kron3(self._identity_phi(), self._identity_theta(), self._varphi_operator())

    def _n_varphi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns the :math:`n_\varphi = - i d/d\\varphi` operator
        """
        return self.varphi_grid.first_derivative_matrix(prefactor=-1j)

    def n_varphi_operator(self):
        """
        Returns
        -------
        ndarray
            Returns charge operator :math:`\\n_varphi` in the total Hilbert space
        """
        return self._kron3(self._identity_phi(), self._identity_theta(), self._n_varphi_operator())

    def _cos_varphi_div_operator(self, div):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\cos \\varphi/div` operator
        """
        cos_varphi_div_vals = np.cos(self.varphi_grid.make_linspace() / div)
        return sparse.dia_matrix((cos_varphi_div_vals, [0]), shape=(self.dim_varphi(), self.dim_varphi())).tocsc()

    def _sin_varphi_div_operator(self, div):
        """
        Returns
        -------
        ndarray
            Returns the :math:`\\sin \\varphi/div` operator
        """
        sin_varphi_div_vals = np.sin(self.varphi_grid.make_linspace() / div)
        return sparse.dia_matrix((sin_varphi_div_vals, [0]), shape=(self.dim_varphi(), self.dim_varphi())).tocsc()

    def _kron3(self, mat1, mat2, mat3):
        """
        Kronecker product of three matrices

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return sparse.kron(sparse.kron(mat1, mat2, format='csc'), mat3, format='csc')

    def _kron2(self, mat1, mat2):
        return sparse.kron(mat1, mat2, format='csc')

    def _identity_phi(self):
        """
        Identity operator acting only on the :math:`\phi` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return sparse.identity(self.dim_phi(), format='csc', dtype=np.complex_)

    def _identity_theta(self):
        """
        Identity operator acting only on the :math:`\theta` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return sparse.identity(self.dim_theta(), format='csc', dtype=np.complex_)

    def _identity_varphi(self):
        """
        Identity operator acting only on the :math:`\varphi` Hilbert subspace.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return sparse.identity(self.dim_varphi(), format='csc', dtype=np.complex_)

    def total_identity(self):
        """
        Identity operator acting only on the total Hilbert space.

        Returns
        -------
            scipy.sparse.csc_mat
        """
        return self._kron3(self._identity_phi(), self._identity_theta(), self._identity_varphi())

    def hamiltonian(self):
        # TODO: why not equal?
        # kinetic_mat = 8 * self.EC * self.n_phi_operator() ** 2 + 2 * self.EC * self.n_varphi_operator() ** 2

        phi_term = self.phi_grid.second_derivative_matrix(prefactor=- 8.0 * self._dis_ec())
        varphi_term = self.varphi_grid.second_derivative_matrix(prefactor=- 2.0 * self._dis_ec())
        kinetic_mat = self._kron3(phi_term, self._identity_theta(), self._identity_varphi()) + self._kron3(
            self._identity_phi(), self._identity_theta(), varphi_term)

        cross_mat = - 4 * self._dis_ec() * self.n_theta_operator() * self.n_varphi_operator()

        theta_osc_mat = self._kron3(self._identity_phi(), op.number_sparse(self.dim_theta(), self.theta_plasma()),
                                    self._identity_varphi()) + 2 * self._dis_ec() * self.n_theta_operator() ** 2

        phi_inductive_mat = 0.25 * self._dis_el() * self.phi_operator() ** 2
        varphi_inductive_mat = 0.5 * self.ELA * self.varphi_operator() ** 2
        add_inductive_mat = self.ELA * (
                self.varphi_operator() * self.theta_operator() + 2 * np.pi * (self.flux / 2.0 + self.fluxa) * (
                self.theta_operator() + self.varphi_operator()))
        phi_flux_term = self._cos_phi_div_operator(2.0) * np.cos(np.pi * self.flux) - self._sin_phi_div_operator(
            2.0) * np.sin(np.pi * self.flux)
        junction_mat = - 2 * self.EJ * self._kron3(phi_flux_term, self._identity_theta(),
                                                   self._cos_varphi_div_operator(
                                                       1.0)) + 2 * self.EJ * self.total_identity()

        return theta_osc_mat + kinetic_mat + cross_mat + phi_inductive_mat + varphi_inductive_mat + add_inductive_mat + junction_mat

    def disorder(self):
        """
        Return disorder Hamiltonian

        Returns
        -------
        ndarray
        """
        disorder_l = - self._dis_el() * self.dL * self._kron3(self._phi_operator(), self._theta_operator(),
                                                              self._identity_varphi())

        phi_flux_term = self._sin_phi_div_operator(2.0) * np.cos(self.flux * np.pi) + self._cos_phi_div_operator(
            2.0) * np.sin(
            self.flux * np.pi)
        disorder_j = 2 * self.EJ * self.dJ * self._kron3(phi_flux_term, self._identity_theta(),
                                                         self._sin_varphi_div_operator(1.0))

        disorder_c = -8 * self._dis_ec() * self.dC * self.n_phi_operator() * (
                self.n_varphi_operator() - self.n_theta_operator())

        return disorder_l + disorder_j + disorder_c

    def potential(self, varphi, phi):
        """
        Double Cooper pair tunneling qubit potential evaluated at `phi, varphi`, with `theta=0`

        Parameters
        ----------
        phi: float or ndarray
            float value of the phase variable `phi`
        varphi: float or ndarray
            float value of the phase variable `varphi`

        Returns
        -------
        float or ndarray
        """
        return self.EL * (0.25 * phi * phi) - 2 * self.EJ * np.cos(varphi) * np.cos(
            phi * 0.5 + np.pi * self.flux) + 0.5 * self.ELA * (
                       2 * np.pi * (self.flux / 2.0 + self.fluxa) + varphi) ** 2 + 2 * self.EJ

    def plot_potential(self, phi_grid=None, varphi_grid=None, contour_vals=None, **kwargs):
        """
        Draw contour plot of the potential energy.

        Parameters
        ----------
        phi_grid: Grid1d, option
            used for setting a custom grid for phi; if None use self._default_phi_grid
        varphi_grid: Grid1d, option
            used for setting a custom grid for varphi; if None use self._default_varphi_grid
        contour_vals: list, optional
        **kwargs:
            plotting parameters
        """
        phi_grid = phi_grid or self.phi_grid
        varphi_grid = varphi_grid or self.varphi_grid

        x_vals = varphi_grid.make_linspace()
        y_vals = phi_grid.make_linspace()
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (4, 4)
        return plot.contours(x_vals, y_vals, self.potential, contour_vals=contour_vals, **kwargs)

    def _evals_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian() + self.disorder()
        evals = eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=False, sigma=0.0, which='LM')
        # evals = eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=False, which='SA')
        return np.sort(evals)

    def _esys_calc(self, evals_count):
        hamiltonian_mat = self.hamiltonian() + self.disorder()
        evals, evecs = eigsh(hamiltonian_mat, k=evals_count, return_eigenvectors=True, sigma=0.0, which='LM')
        evals, evecs = spec_utils.order_eigensystem(evals, evecs)
        return evals, evecs

    def wavefunction(self, esys=None, which=0, phi_grid=None, theta_grid=None, varphi_grid=None):
        evals_count = max(which + 1, 3)
        if esys is None:
            _, evecs = self.eigensys(evals_count)
        else:
            _, evecs = esys

        phi_grid = phi_grid or self.phi_grid
        theta_grid = theta_grid or self.theta_grid
        varphi_grid = varphi_grid or self.varphi_grid

        state_amplitudes = evecs[:, which].reshape(self.dim_phi(), self.dim_theta(), self.dim_varphi())

        theta_osc_amplitudes = np.zeros((self.dim_theta(), theta_grid.pt_count), dtype=np.complex_)
        for i in range(self.dim_theta()):
            theta_osc_amplitudes[i, :] = osc.harm_osc_wavefunction(i, theta_grid.make_linspace(), self.theta_osc())

        wavefunc_amplitudes = np.swapaxes(np.tensordot(theta_osc_amplitudes, state_amplitudes, axes=([0], [1])), 0, 1)
        wavefunc_amplitudes = spec_utils.standardize_phases(wavefunc_amplitudes)

        grid3d = discretization.GridSpec(
            np.asarray([[phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count],
                        [theta_grid.min_val, theta_grid.max_val, theta_grid.pt_count],
                        [varphi_grid.min_val, varphi_grid.max_val, varphi_grid.pt_count]]))
        return storage.WaveFunctionOnGrid(grid3d, wavefunc_amplitudes)

    def plot_phi_varphi_wavefunction(self, esys=None, which=0, phi_grid=None, varphi_grid=None, mode='abs',
                                     zero_calibrate=True,
                                     **kwargs):
        """
        Plots 2D phase-basis wave function at theta = 0

        Parameters
        ----------
        esys: ndarray, ndarray
            eigenvalues, eigenvectors as obtained from `.eigensystem()`
        which: int, optional
            index of wave function to be plotted (default value = (0)
        phi_grid: Grid1d, option
            used for setting a custom grid for phi; if None use self._default_phi_grid
        varphi_grid: Grid1d, option
            used for setting a custom grid for varphi; if None use self._default_varphi_grid
        mode: str, optional
            choices as specified in `constants.MODE_FUNC_DICT` (default value = 'abs_sqr')
        zero_calibrate: bool, optional
            if True, colors are adjusted to use zero wavefunction amplitude as the neutral color in the palette
        **kwargs:
            plot options

        Returns
        -------
        Figure, Axes
        """
        phi_grid = phi_grid or self.phi_grid
        theta_grid = discretization.Grid1d(0, 0, 1)
        varphi_grid = varphi_grid or self.varphi_grid

        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, theta_grid=theta_grid, varphi_grid=varphi_grid,
                                     which=which)

        wavefunc.gridspec = discretization.GridSpec(np.asarray(
            [[varphi_grid.min_val, varphi_grid.max_val, varphi_grid.pt_count],
             [phi_grid.min_val, phi_grid.max_val, phi_grid.pt_count]]))
        wavefunc.amplitudes = amplitude_modifier(
            spec_utils.standardize_phases(wavefunc.amplitudes.reshape(phi_grid.pt_count, varphi_grid.pt_count)))
        return plot.wavefunction2d(wavefunc, zero_calibrate=zero_calibrate, **kwargs)

    def plot_n_phi_n_varphi_wavefunction(self, esys=None, mode='real', which=0, zero_calibrate=True, **kwargs):
        phi_grid = self.phi_grid
        theta_grid = discretization.Grid1d(0, 0, 1)
        varphi_grid = self.varphi_grid

        wavefunc = self.wavefunction(esys, phi_grid=phi_grid, theta_grid=theta_grid, varphi_grid=varphi_grid,
                                     which=which)

        amplitudes = spec_utils.standardize_phases(
            wavefunc.amplitudes.reshape(phi_grid.pt_count, varphi_grid.pt_count))

        d_phi = phi_grid.make_linspace()[1] - phi_grid.make_linspace()[0]
        n_phi_list = np.sort(np.fft.fftfreq(phi_grid.pt_count, d_phi)) * 2 * np.pi
        n_phi_grid = discretization.Grid1d(n_phi_list[0], n_phi_list[-1], n_phi_list.size)

        d_varphi = varphi_grid.make_linspace()[1] - varphi_grid.make_linspace()[0]
        n_varphi_list = np.sort(np.fft.fftfreq(varphi_grid.pt_count, d_varphi)) * 2 * np.pi
        n_varphi_grid = discretization.Grid1d(n_varphi_list[0], n_varphi_list[-1], n_varphi_list.size)

        n_phi_n_varphi_amplitudes = np.fft.ifft2(
            amplitudes) * d_phi * phi_grid.pt_count * d_varphi * varphi_grid.pt_count
        n_phi_n_varphi_amplitudes = np.fft.fftshift(n_phi_n_varphi_amplitudes)

        grid2d = discretization.GridSpec(np.asarray([
            [n_phi_grid.min_val, n_phi_grid.max_val, n_phi_grid.pt_count],
            [n_varphi_grid.min_val, n_varphi_grid.max_val, n_varphi_grid.pt_count]]))

        n_phi_n_varphi_wavefunction = storage.WaveFunctionOnGrid(grid2d, n_phi_n_varphi_amplitudes)
        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        n_phi_n_varphi_wavefunction.amplitudes = amplitude_modifier(
            spec_utils.standardize_phases(n_phi_n_varphi_wavefunction.amplitudes))

        return plot.wavefunction2d(n_phi_n_varphi_wavefunction, zero_calibrate=zero_calibrate, **kwargs)

    def phi_1_operator(self):
        """
        phase drop on inductor 1
        used in inductive loss calculation
        """
        return self.theta_operator() - self.phi_operator() / 2.0

    def phi_2_operator(self):
        """
        phase drop on indcutor 2
        used in inductive loss calculation
        """
        return - self.theta_operator() - self.phi_operator() / 2.0

    def phi_a_operator(self):
        """
        phase drop on additional inductor
        used in inductive loss calculation
        """
        return self.theta_operator() + self.varphi_operator()

    def q_ind(self, energy):
        """Frequency dependent quality factor of inductance"""
        q_ind_0 = 500 * 1e6
        return q_ind_0 * kn(0, 0.5 / 2.0 / self.kbt) * np.sinh(0.5 / 2.0 / self.kbt) / kn(0,
                                                                                          energy / 2.0 / self.kbt) / np.sinh(
            energy / 2.0 / self.kbt)

    def N_1_operator(self):
        """
        charge across junction 1
        used in capacitive loss calculation
        """
        return self.n_phi_operator() + 0.5 * (self.n_varphi_operator() - self.n_theta_operator())

    def N_2_operator(self):
        """
        charge across junction 2
        used in capacitive loss calculation
        """
        return self.n_phi_operator() - 0.5 * (self.n_varphi_operator() - self.n_theta_operator())

    def sin_varphi_1_2_operator(self):
        """sin(\varphi_1/2)"""
        cos_phi_4 = self._kron3(self._cos_phi_div_operator(4.0), self._identity_theta(), self._identity_varphi())
        sin_phi_4 = self._kron3(self._sin_phi_div_operator(4.0), self._identity_theta(), self._identity_varphi())
        cos_varphi_2 = self._kron3(self._identity_phi(), self._identity_theta(), self._cos_varphi_div_operator(2.0))
        sin_varphi_2 = self._kron3(self._identity_phi(), self._identity_theta(), self._sin_varphi_div_operator(2.0))

        return sin_phi_4 * cos_varphi_2 + cos_phi_4 * sin_varphi_2

    def sin_varphi_2_2_operator(self):
        """sin(\varphi_2/2)"""
        cos_phi_4 = self._kron3(self._cos_phi_div_operator(4.0), self._identity_theta(), self._identity_varphi())
        sin_phi_4 = self._kron3(self._sin_phi_div_operator(4.0), self._identity_theta(), self._identity_varphi())
        cos_varphi_2 = self._kron3(self._identity_phi(), self._identity_theta(), self._cos_varphi_div_operator(2.0))
        sin_varphi_2 = self._kron3(self._identity_phi(), self._identity_theta(), self._sin_varphi_div_operator(2.0))

        return sin_phi_4 * cos_varphi_2 - cos_phi_4 * sin_varphi_2

    def y_qp(self, energy):
        """frequency dependent addimitance for quasiparticle"""
        gap = 80.0
        xqp = 1e-8
        return 16 * np.pi * np.sqrt(2 / np.pi) / gap * energy * (2 * gap / energy) ** 1.5 * xqp * np.sqrt(
            energy / 2 / self.kbt) * kn(0, energy / 2 / self.kbt) * np.sinh(energy / 2 / self.kbt)

    def q_cap(self, energy):
        """Frequency dependent quality factor of capacitance"""

        # Devoret paper
        q_cap_0 = 1 * 1e6
        return q_cap_0 * (6 / energy) ** 0.7

        # Schuster paper
        # return 1 / (8e-6)

        # Vlad paper
        # q_cap_0 = 1 / (3 * 1e-6)
        # return q_cap_0 * (6 / energy) ** 0.15

    def thermal_factor(self, energy):
        return np.where(energy > 0, 0.5 * (1 / (np.tanh(energy / 2.0 / self.kbt)) + 1),
                        0.5 * (1 / (np.tanh(- energy / 2.0 / self.kbt)) - 1))

    def get_t1_capacitive_loss(self, init_state):
        """T1 capacitive loss of one particular state"""
        cutoff = init_state + 4
        energy = self._evals_calc(cutoff)
        energy_diff = energy[init_state] - energy
        energy_diff = np.delete(energy_diff, init_state)

        matelem_1 = self.get_matelements_vs_paramvals('N_1_operator', 'ph', [0], evals_count=cutoff).matrixelem_table[0,
                    init_state, :]
        matelem_1 = np.delete(matelem_1, init_state)
        matelem_2 = self.get_matelements_vs_paramvals('N_2_operator', 'ph', [0], evals_count=cutoff).matrixelem_table[0,
                    init_state, :]
        matelem_2 = np.delete(matelem_2, init_state)

        s_vv_1 = 2 * np.pi * 16 * self.EC / (1 - self.dC ** 2) / self.q_cap(np.abs(energy_diff)) * self.thermal_factor(
            energy_diff)
        s_vv_2 = 2 * np.pi * 16 * self.EC / (1 + self.dC ** 2) / self.q_cap(np.abs(energy_diff)) * self.thermal_factor(
            energy_diff)

        gamma1_cap_1 = np.abs(matelem_1) ** 2 * s_vv_1
        gamma1_cap_2 = np.abs(matelem_2) ** 2 * s_vv_2

        gamma1_cap_tot = np.sum(gamma1_cap_1) + np.sum(gamma1_cap_2)
        return 1 / (gamma1_cap_tot) * 1e-6

    def get_t1_purcell(self, init_state):
        """T1 purcell loss of one particular state"""
        cutoff = init_state + 4
        energy = self._evals_calc(cutoff)
        energy_diff = energy[init_state] - energy
        energy_diff = np.delete(energy_diff, init_state)

        matelem = self.get_matelements_vs_paramvals('n_theta_operator', 'ph', [0], evals_count=cutoff).matrixelem_table[
                  0, init_state, :]
        matelem = np.delete(matelem, init_state)

        # note here only matters the shunt capacitance, so EC*x
        s_vv = 2 * np.pi * 16 * self.EC * self.x / self.q_cap(np.abs(energy_diff)) * self.thermal_factor(energy_diff)

        gamma1_purcell = np.abs(matelem) ** 2 * s_vv

        gamma1_purcell_tot = np.sum(gamma1_purcell)
        return 1 / (gamma1_purcell_tot) * 1e-6

    # TODO: check the factor in front of addtional inductor in the spectral density
    def get_t1_inductive_loss(self, init_state):
        """T1 inductive loss of one particular state"""
        cutoff = init_state + 4
        energy = self._evals_calc(cutoff)
        energy_diff = energy[init_state] - energy
        energy_diff = np.delete(energy_diff, init_state)

        matelem_1 = self.get_matelements_vs_paramvals('phi_1_operator', 'ph', [0], evals_count=cutoff).matrixelem_table[
                    0, init_state, :]
        matelem_1 = np.delete(matelem_1, init_state)
        matelem_2 = self.get_matelements_vs_paramvals('phi_2_operator', 'ph', [0], evals_count=cutoff).matrixelem_table[
                    0, init_state, :]
        matelem_2 = np.delete(matelem_2, init_state)
        matelem_a = self.get_matelements_vs_paramvals('phi_a_operator', 'ph', [0], evals_count=cutoff).matrixelem_table[
                    0, init_state, :]
        matelem_a = np.delete(matelem_a, init_state)

        s_ii_1 = 2 * np.pi * 2 * self.EL / (1 - self.dL ** 2) / self.q_ind(np.abs(energy_diff)) * self.thermal_factor(
            energy_diff)
        s_ii_2 = 2 * np.pi * 2 * self.EL / (1 + self.dL ** 2) / self.q_ind(np.abs(energy_diff)) * self.thermal_factor(
            energy_diff)
        s_ii_a = 2 * np.pi * 2 * self.ELA / self.q_ind(np.abs(energy_diff)) * self.thermal_factor(energy_diff)

        gamma1_ind_1 = np.abs(matelem_1) ** 2 * s_ii_1
        gamma1_ind_2 = np.abs(matelem_2) ** 2 * s_ii_2
        gamma1_ind_a = np.abs(matelem_a) ** 2 * s_ii_a

        gamma1_ind_tot = np.sum(gamma1_ind_1) + np.sum(gamma1_ind_2) + np.sum(gamma1_ind_a)
        return 1 / (gamma1_ind_tot) * 1e-6

    def get_t1_inductive_loss_channel(self, init_state):
        """T1 inductive loss of one particular state"""
        cutoff = init_state + 6
        energy = self._evals_calc(cutoff)
        energy_diff = energy[init_state] - energy
        energy_diff = np.delete(energy_diff, init_state)

        matelem_1 = self.get_matelements_vs_paramvals('phi_1_operator', 'ph', [0], evals_count=cutoff).matrixelem_table[
                    0, init_state, :]
        matelem_1 = np.delete(matelem_1, init_state)
        matelem_2 = self.get_matelements_vs_paramvals('phi_2_operator', 'ph', [0], evals_count=cutoff).matrixelem_table[
                    0, init_state, :]
        matelem_2 = np.delete(matelem_2, init_state)
        matelem_a = self.get_matelements_vs_paramvals('phi_a_operator', 'ph', [0], evals_count=cutoff).matrixelem_table[
                    0, init_state, :]
        matelem_a = np.delete(matelem_a, init_state)

        s_ii_1 = 2 * np.pi * 2 * self.EL / (1 - self.dL ** 2) / self.q_ind(np.abs(energy_diff)) * self.thermal_factor(
            energy_diff)
        s_ii_2 = 2 * np.pi * 2 * self.EL / (1 + self.dL ** 2) / self.q_ind(np.abs(energy_diff)) * self.thermal_factor(
            energy_diff)
        s_ii_a = 2 * np.pi * 2 * self.ELA / self.q_ind(np.abs(energy_diff)) * self.thermal_factor(energy_diff)

        gamma1_ind_1 = np.abs(matelem_1) ** 2 * s_ii_1
        gamma1_ind_2 = np.abs(matelem_2) ** 2 * s_ii_2
        gamma1_ind_a = np.abs(matelem_a) ** 2 * s_ii_a

        gamma1_ind_tot = np.sum(gamma1_ind_1) + np.sum(gamma1_ind_2) + np.sum(gamma1_ind_a)
        gamma_channel = gamma1_ind_1 + gamma1_ind_2 + gamma1_ind_a
        return 1 / gamma_channel * 1e-6

    def get_t1_qp_loss(self, init_state):
        """T1 quasiparticle loss of one particular state"""
        cutoff = init_state + 4
        energy = self._evals_calc(cutoff)
        energy_diff = energy[init_state] - energy
        energy_diff = np.delete(energy_diff, init_state)

        matelem_1 = self.get_matelements_vs_paramvals('sin_varphi_1_2_operator', 'ph', [0],
                                                      evals_count=cutoff).matrixelem_table[
                    0, init_state, :]
        matelem_1 = np.delete(matelem_1, init_state)
        matelem_2 = self.get_matelements_vs_paramvals('sin_varphi_2_2_operator', 'ph', [0],
                                                      evals_count=cutoff).matrixelem_table[
                    0, init_state, :]
        matelem_2 = np.delete(matelem_2, init_state)

        s_qp_1 = self.EJ * (1 - self.dJ) * self.y_qp(np.abs(energy_diff)) * self.thermal_factor(energy_diff)
        s_qp_2 = self.EJ * (1 + self.dJ) * self.y_qp(np.abs(energy_diff)) * self.thermal_factor(energy_diff)

        gamma1_qp_1 = np.abs(matelem_1) ** 2 * s_qp_1
        gamma1_qp_2 = np.abs(matelem_2) ** 2 * s_qp_2

        gamma1_ind_tot = np.sum(gamma1_qp_1) + np.sum(gamma1_qp_2)
        return 1 / (gamma1_ind_tot) * 1e-6

    def get_t2_flux_noise(self, init_state):
        delta = 1e-6
        pts = 21
        flux_list = np.linspace(self.flux - delta, self.flux + delta, pts)
        energy = self.get_spectrum_vs_paramvals('flux', flux_list, evals_count=init_state + 2,
                                                subtract_ground=True).energy_table[:, init_state]
        first_derivative = np.gradient(energy, flux_list)[int(np.round(pts / 2))]
        second_derivative = np.gradient(np.gradient(energy, flux_list), flux_list)[int(np.round(pts / 2))]

        first_order = 3e-6 * np.abs(first_derivative)
        second_order = 9e-12 * np.abs(second_derivative)
        return np.abs(1 / (first_order + second_order) * 1e-6) / (2 * np.pi)  # unit in ms

    def get_t2_fluxa_noise(self, init_state):
        delta = 1e-6
        pts = 21
        flux_list = np.linspace(self.fluxa - delta, self.fluxa + delta, pts)
        energy = self.get_spectrum_vs_paramvals('fluxa', flux_list, evals_count=init_state + 2,
                                                subtract_ground=True).energy_table[:, init_state]
        first_derivative = np.gradient(energy, flux_list)[int(np.round(pts / 2))]
        second_derivative = np.gradient(np.gradient(energy, flux_list), flux_list)[int(np.round(pts / 2))]

        first_order = 3e-6 * np.abs(first_derivative)
        second_order = 9e-12 * np.abs(second_derivative)
        return np.abs(1 / (first_order + second_order) * 1e-6) / (2 * np.pi)  # unit in ms

    def get_t2_current_noise(self, init_state):
        delta = 1e-7
        pts = 21
        ej_list = np.linspace(self.EJ - delta, self.EJ + delta, pts)
        energy = self.get_spectrum_vs_paramvals('EJ', ej_list, evals_count=init_state + 2,
                                                subtract_ground=True).energy_table[:, init_state]
        first_derivative = np.gradient(energy, ej_list)[int(np.round(pts / 2))]
        # assume EJ here is the average EJ in the presence of disorder
        return np.abs(1 / (5e-7 * self.EJ * np.abs(first_derivative)) * 1e-6) / (2 * np.pi)  # unit in ms

    def print_noise(self, g_state, e_state):
        t2_current = self.get_t2_current_noise(e_state)
        t2_flux = self.get_t2_flux_noise(e_state)
        t2_fluxa = self.get_t2_fluxa_noise(e_state)
        t1_cap = 1 / (1 / self.get_t1_capacitive_loss(g_state) + 1 / self.get_t1_capacitive_loss(e_state))
        t1_ind = 1 / (1 / self.get_t1_inductive_loss(g_state) + 1 / self.get_t1_inductive_loss(e_state))
        t1_qp = 1 / (1 / self.get_t1_qp_loss(g_state) + 1 / self.get_t1_qp_loss(e_state))
        t1_purcell = 1 / (1 / self.get_t1_purcell(g_state) + 1 / self.get_t1_purcell(e_state))
        t1_tot = 1 / (1 / t1_cap + 1 / t1_ind + 1 / t1_purcell + 1 / t1_qp)
        t2_tot = 1 / (1 / t2_current + 1 / t2_flux + 1 / t2_fluxa + 1 / t1_tot / 2)

        return print(' T2_current =', t2_current, ' ms', '\n T2_flux =', t2_flux,
                     ' ms', '\n T2_flux_a =', t2_fluxa,
                     ' ms', '\n T1_cap =',
                     t1_cap, ' ms', '\n T1_Purcell =',
                     t1_purcell, ' ms', '\n T1_ind =', t1_ind, ' ms', '\n T1_qp =', t1_qp, ' ms', '\n T1 =', t1_tot,
                     ' ms', '\n T2 =', t2_tot,
                     ' ms')

    def d_ham_d_flux_operator(self):
        return - self.EL * (0.5 * (self.phi_operator() - self.total_identity() * 2 * np.pi * self.flux) + (
                self.theta_operator() - self.total_identity() * 2 * np.pi * (self.flux / 2.0 + self.fluxa)))

    def d_ham_d_fluxa_operator(self):
        return - self.EL * 2.0 * (
                self.theta_operator() - self.total_identity() * 2 * np.pi * (self.flux / 2.0 + self.fluxa))

    def d_ham_d_flux2_operator(self):
        return - self.EL * (0.5 * (self.phi_operator() - self.total_identity() * 2 * np.pi * self.flux) - (
                self.theta_operator() - self.total_identity() * 2 * np.pi * (self.flux / 2.0 + self.fluxa)))

    def d_ham_d_fluxc_operator(self):
        return - self.EL * (self.phi_operator() - self.total_identity() * 2 * np.pi * self.flux)

    def d_ham_d_fluxd_operator(self):
        return - 2 * self.EL * (self.theta_operator() - self.total_identity() * 2 * np.pi * (self.flux / 2.0 + self.fluxa))

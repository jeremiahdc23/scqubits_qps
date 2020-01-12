# plot_defaults.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################


import numpy as np

from scqubits.settings import DEFAULT_ENERGY_UNITS


def wavefunction1d():
    """Plot defaults for plotting.wavefunction1d"""
    return {
        'xlabel': r'$\varphi$',
        'ylabel': r'$\psi_j(\varphi)$'
    }


def wavefunction1d_discrete():
    """Plot defaults for plotting.wavefunction1d_discrete"""
    return {
        'xlabel': 'n',
        'ylabel': r'$\psi_j(n)$'
    }


def wavefunction2d():
    """Plot defaults for plotting.wavefunction2d"""
    return {
        'figsize': (8, 3)
    }


def contours(x_vals, y_vals):
    """Plot defaults for plotting.contours"""
    aspect_ratio = (y_vals[-1] - y_vals[0]) / (x_vals[-1] - x_vals[0])
    figsize = (8, 8 * aspect_ratio)
    return {
        'figsize': figsize
    }


def matrix():
    """Plot defaults for plotting.matrix"""
    return {
        'figsize': (10, 5)
    }


def evals_vs_paramvals(specdata, **kwargs):
    """Plot defaults for plotting.evals_vs_paramvals"""
    kwargs['xlabel'] = kwargs.get('xlabel') or specdata.param_name
    kwargs['ylabel'] = kwargs.get('ylabel') or 'energy [{}]'.format(DEFAULT_ENERGY_UNITS)
    return kwargs


def matelem_vs_paramvals(specdata):
    """Plot defaults for plotting.matelem_vs_paramvals"""
    return {
        'xlabel': specdata.param_name,
        'ylabel': 'matrix element'
    }


def dressed_spectrum(sweep, **kwargs):
    """Plot defaults for sweep_plotting.dressed_spectrum"""
    kwargs['ymax'] = kwargs.get('ymax') or min(15, (np.max(sweep.dressed_specdata.energy_table) -
                                                    np.min(sweep.dressed_specdata.energy_table)))
    return kwargs


def chi(sweep, **kwargs):
    """Plot defaults for sweep_plotting.chi"""
    kwargs['xlabel'] = kwargs.get('xlabel') or sweep.param_name
    kwargs['ylabel'] = kwargs.get('ylabel') or r'$\chi_j$' + DEFAULT_ENERGY_UNITS
    return kwargs


def chi01(sweep, yval, **kwargs):
    """Plot defaults for sweep_plotting.chi01"""
    kwargs['xlabel'] = kwargs.get('xlabel') or sweep.param_name
    kwargs['ylabel'] = kwargs.get('ylabel') or r'$\chi_{{01}}$ [{}]'.format(DEFAULT_ENERGY_UNITS)
    kwargs['title'] = kwargs.get('title') or r'$\chi_{{01}}=${:.4f} {}'.format(yval, DEFAULT_ENERGY_UNITS)
    return kwargs


def charge_matrixelem(sweep, **kwargs):
    """Plot defaults for sweep_plotting.charge_matrixelem"""
    kwargs['xlabel'] = kwargs.get('xlabel') or sweep.param_name
    kwargs['ylabel'] = kwargs.get('ylabel') or r'$|\langle i |n| j \rangle|$'
    return kwargs

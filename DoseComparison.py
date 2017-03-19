#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
DoseComparison
    numerical evaluation of dose distributions comparison

:Date: 2017-01-02
:Version: 2.0.0
:Author: ongchi
:Copyright: Copyright (c) 2016, ongchi
:License: BSD 3-Clause License
'''

from collections import namedtuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

VolumeImage = namedtuple('VolumeImage', 'x y z v')


class DoseComparison(object):
    ''' Radiation Dose Distribution Comparisons
    '''
    def __init__(self, reference, test, delta_d=0.05, delta_r=3):
        super().__init__()

        self._delta_d = delta_d
        self._delta_r = delta_r

        self._voxel_size, self._span, \
        self._axis, self._ext_axis, \
        self._grid, self._ext_grid, \
        self._shape, self._ext_shape = self._prepare_ax_grid(reference, test, delta_r)

        self._ref, self._test, self._ext_test = self._prepare_volume_data(reference, test)

        self._slab_idx_list = list(map(

            lambda f, t: (f, t),
            range(len(self._test)),
            range(len(self._ext_test) - len(self._test), len(self._ext_test))
        ))
        self._r_list = self._pts_in_delta_r()

        Result = namedtuple('Result', 'isDone value')
        self._dd = Result(False, np.empty_like(self._ref))
        self._gamma = Result(False, np.full_like(self._ref, np.inf))
        self._ndd = Result(False, np.empty_like(self._ref))

    def get_gamma(self):
        if not self._gamma.isDone:
            for i, _slab in enumerate(self._slab_idx_list):
                gamma = self._gamma.value[i]
                for _slice, r in self._r_list:
                    tmp_gamma = self._gamma_from_r(
                        r,
                        self.get_dd()[i],
                        self._ref[i],
                        self._ext_test[slice(_slab)][_slice]
                    )
                    _gt = gamma > tmp_gamma
                    gamma[_gt] = tmp_gamma[_gt]
            self._gamma = self._gamma._replace(isDone=True)
        return self._gamma.value

    def get_dd(self):
        if not self._dd.isDone:
            self._dd.value[:] = self._test - self._ref
            self._dd = self._dd._replace(idDone=True)
        return self._dd.value

    def _prepare_ax_grid(self, r, t, delta_r):
        min_voxel_size = [
            np.hstack([np.gradient(r.x), np.gradient(t.x)]).min(),
            np.hstack([np.gradient(r.y), np.gradient(t.y)]).min(),
            np.hstack([np.gradient(r.z), np.gradient(t.z)]).min()
        ]

        ax_stack = [np.hstack([r.x, t.x]), np.hstack([r.y, t.y]), np.hstack([r.z, t.z])]

        left, right = np.array([i.min() for i in ax_stack]), np.array([i.max() for i in ax_stack])

        axis = self._ax_gen(left, right, min_voxel_size)

        voxel_size = np.array([i[1] - i[0] for i in axis])
        ext_scale_factor = np.ceil([delta_r / sz for sz in voxel_size])
        span = voxel_size * ext_scale_factor

        ext_axis = self._ax_gen(left - span, right + span, voxel_size)

        grid = np.array(np.meshgrid(*axis, indexing='ij'))
        ext_grid = np.array(np.meshgrid(*ext_axis, indexing='ij'))
        shape = [len(i) for i in axis]
        ext_shape = [len(i) for i in ext_axis]

        return voxel_size, span, axis, ext_axis, grid, ext_grid, shape, ext_shape

    def _prepare_volume_data(self, r, t):
        ref_interp_f = self._structured_griddata_interp(r)
        test_interp_f = self._structured_griddata_interp(t)

        ref = ref_interp_f(self._grid.T.reshape(-1, 3)).reshape(*self._shape)
        test = test_interp_f(self._grid.T.reshape(-1, 3)).reshape(*self._shape)
        ext_test = test_interp_f(self._ext_grid.T.reshape(-1, 3)).reshape(*self._ext_shape)

        return ref, test, ext_test

    def _pts_in_delta_r(self):
        r_map = np.sqrt(np.sum(np.square(np.meshgrid(
            *self._ax_gen(-self._span, self._span, self._voxel_size),
            indexing='ij'
        )), axis=0))
        r_map[r_map > self._delta_r] = np.nan

        ni, nj, nk = self._shape

        r_list = []
        iter_r = np.nditer(r_map, ("multi_index", ))
        while not iter_r.finished:
            i, j, k = idx = iter_r.multi_index
            if not np.isnan(r_map[idx]):
                r_list.append((
                    [slice(i, i + ni), slice(j, j + nj), slice(k, k + nk)],
                    r_map[idx]
                ))
            iter_r.iternext()

        return r_list

    @staticmethod
    def _ax_gen(left, right, size):
        return list(map(
            lambda l, r, s: np.linspace(l, r, np.ceil((r - l) / s + 1).astype(int)),
            left, right, size
        ))

    @staticmethod
    def _structured_griddata_interp(volume):
        grid_interp = RegularGridInterpolator(
            (volume.x, volume.y, volume.z),
            volume.v,
            method='linear',
            bounds_error=False,
            fill_value=0
        )

        return grid_interp

    @staticmethod
    def _shared_array(np_arr):
        return np.ndarray(np_arr.shape, dtype=np_arr.dtype, buffer=np_arr.data)

    def _gamma_from_r(self, r, dd, ref, test):
        delta_d, delta_r = self._delta_d, self._delta_r
        r_sq = (r / delta_r) ** 2
        d_sq = (dd / ref / delta_d) ** 2
        return np.sqrt(r_sq + d_sq)

    def _madd_ndd(self, d_min, r_min, dd, ref, test):
        delta_d, delta_r = self._delta_d, self._delta_r
        sr = np.sqrt(delta_d ** 2 - (delta_d * r_min / delta_r) ** 2)
        dd_min_abs = np.abs(d_min - ref)
        in_sr = ((dd_min_abs / ref) < sr)
        madd = in_sr * (sr - dd_min_abs + delta_d)
        madd[madd < delta_d] = delta_d
        ndd = dd / madd * delta_d
        return madd, ndd

if __name__ == '__main__':
    def wave(x, y, z, v):
        return np.cos(v * x) + np.cos(v * y) + np.cos(v * z) \
            + 2 * (x ** 2 + y ** 2 + z ** 2)

    x = np.linspace(-2, 2, 5)
    y = np.linspace(-2, 2, 10)
    z = np.linspace(-2, 2, 10)
    mgrid = np.meshgrid(x, y, z, indexing='ij')

    img1 = VolumeImage(x, y, z, wave(*mgrid, 10))
    img2 = VolumeImage(x+0.2, y+0.2, z+0.2, wave(*mgrid, 12))

    comp = DoseComparison(img1, img2, delta_r=0.5, delta_d=0.1)
    comp.get_dd()

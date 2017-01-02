#!/usr/bin/env python
'''
DoseComparison
    numerical evaluation of dose distributions comparison

:Date: 2017-01-02
:Version: 1.0.0
:Author: ongchi
:Copyright: Copyright (c) 2016, ongchi
:License: BSD 3-Clause License
'''
import numpy as np


def DoseComparison(refimg, tstimg, delta_r=1, delta_d=0.05):
    ''' gamma evaluation and normalized dose difference of dose image distributions

    :param refimg: reference dose image
    :param tstimg: test dose image
    :param delta_r: spatial criterion (voxels)
    :param delta_d: dose criterion (percentage)

    :type refimg: numpy.ndarray
    :type tstimg: numpy.ndarray
    :type delta_r: int
    :type delta_d: float
    :rtype: numpy.ndarray, numpy.ndarray
    '''
    # check for valid arguments
    if refimg.shape != tstimg.shape:
        raise Exception("ValueError: shape mismatch: refimg and tstimg must have the same shape")
    if delta_r <= 0 or int(delta_r) != delta_r:
        raise Exception("ValueError: delta_r is an integer greater than zero")
    if delta_d <= 0 or delta_d >= 1:
        raise Exception("ValueError: delta_d is a float number between 0 (exclusive) and 1 (exclusive)")

    diff = tstimg - refimg

    _ = np.empty(np.array(tstimg.shape) + delta_r * 2)
    exTest = np.ma.array(_, mask=np.ones_like(_, dtype=bool))
    _ = slice(delta_r, -delta_r)
    exTest.data[_, _, _], exTest.mask[_, _, _] = tstimg, False

    # distance map
    distRange = np.arange(-delta_r, delta_r + 1, 1, dtype=float)
    _ = np.array(np.meshgrid(distRange, distRange, distRange))
    distMap = np.sqrt(np.sum(_ ** 2, axis=0))

    # mask out distance map if center of vexel is in delta_r
    distRange[distRange < 0] += 0.5
    distRange[distRange > 0] -= 0.5
    _ = np.array(np.meshgrid(distRange, distRange, distRange))
    distMask = np.sqrt(np.sum(_ ** 2, axis=0)) >= delta_r

    # mask distance within delta_r
    dist = np.ma.array(distMap, mask=distMask)
    dist[dist > delta_r] = delta_r

    # gamma
    gamma = np.ma.empty_like(diff)
    gamma[:] = np.inf
    _sqDist = (dist / delta_r) ** 2

    # normalized dose difference
    madd = np.ma.empty_like(diff)
    madd[:] = -np.inf
    l_min_dose = np.ma.empty_like(diff)
    l_min_dose[:] = np.inf
    l_min_dist = np.ma.empty_like(diff)

    nx, ny, nz = diff.shape
    it = np.nditer(dist, ("multi_index", ))
    while not it.finished:
        i, j, k = idx = it.multi_index
        _volSlice = [slice(i, i + nx), slice(j, j + ny), slice(k, k + nz)]

        # skip masked voxels
        if distMask[idx] or np.alltrue(exTest[_volSlice].mask):
            it.iternext()
            continue

        # gamma index
        _sqDose = ((exTest[_volSlice] - refimg) / refimg / delta_d) ** 2
        _gamma = np.sqrt(_sqDist[idx] + _sqDose)
        _ = np.bitwise_and(gamma > _gamma, np.bitwise_not(_gamma.mask))
        gamma[_] = _gamma[_]

        # madd
        _ = l_min_dose > exTest[_volSlice]
        l_min_dose[_] = exTest[_]
        l_min_dist[_] = dist[idx]

        it.iternext()

    # ndd calculation
    sr = np.sqrt(delta_d ** 2 - (delta_d * l_min_dist / delta_r) ** 2)
    madd = ( (np.abs(l_min_dose - refimg) / refimg) < sr
        ) * ( sr - np.abs(l_min_dose - refimg) + delta_d )
    madd[madd < delta_d] = delta_d
    ndd = diff / madd * delta_d

    return gamma, ndd


if __name__ == '__main__':
    def wave(x, y, z, v):
        return np.cos(v * x) + np.cos(v * y) + np.cos(v * z) \
            + 2 * (x ** 2 + y ** 2 + z ** 2)

    X, Y, Z = np.mgrid[-2:2:500j, -2:2:500j, -2:2:5j]
    img1 = wave(X, Y, Z, 10)
    img2 = wave(X, Y, Z, 12)

    gamma, ndd = DoseComparison(img1, img2, delta_r=2, delta_d=0.1)

#!/usr/bin/env python
'''
DoseComparison
    numerical evaluation of dose distributions comparison

:Date: 2016-11-06
:Version: 0.0.2
:Author: ongchi
:Copyright: Copyright (c) 2016, ongchi
:License: BSD 3-Clause License
'''
import numpy as np


def gamma_index(refimg, tstimg, dta=1, dd=0.05):
    ''' gamma evaluation of dose image distributions

    :param refimg: reference dose image
    :param tstimg: test dose image
    :param dta: distance-to-agreement criterion (voxels)
    :param dd: dose difference criterion

    :type refimg: numpy.ndarray
    :type tstimg: numpy.ndarray
    :type dta: int
    :type dd: float
    :rtype: numpy.ndarray
    '''
    # check for valid arguments
    if refimg.shape != tstimg.shape:
        raise Exception("ValueError: shape mismatch: refimg and tstimg must have the same shape")
    if dta <= 0 or int(dta) != dta:
        raise Exception("ValueError: dta is an integer greater than zero")
    if dd <= 0 or dd >= 1:
        raise Exception("ValueError: dd is a float number between 0 (exclusive) and 1 (exclusive)")

    diff = tstimg - refimg

    gamma = np.ma.empty_like(diff)
    gamma[:] = np.inf

    tmp = np.empty(np.array(tstimg.shape) + dta * 2)
    exTest = np.ma.array(tmp, mask=np.ones_like(tmp, dtype=bool))
    exTest.data[dta:-dta, dta:-dta, dta:-dta] = tstimg
    exTest.mask[dta:-dta, dta:-dta, dta:-dta] = False

    # distance grid
    distRange = np.arange(-dta, dta + 1, 1, dtype=float)
    tmp = np.array(np.meshgrid(distRange, distRange, distRange))
    distGrid = np.sqrt(np.sum(tmp ** 2, axis=0))

    # mask of distance grid
    distRange[distRange < 0] += 0.5
    distRange[distRange > 0] -= 0.5
    tmp = np.array(np.meshgrid(distRange, distRange, distRange))
    distMask = np.sqrt(np.sum(tmp ** 2, axis=0)) >= dta

    # masked distance within dta
    dist = np.ma.array(distGrid, mask=distMask)
    dist[dist > dta] = dta

    # gamma evaluation
    nz, ny, nx = diff.shape
    _sqDTA = (dist / dta) ** 2
    it = np.nditer(dist, ("multi_index", ))
    while not it.finished:
        i, j, k = idx = it.multi_index
        _volSlice = [slice(i, i + nz), slice(j, j + ny), slice(k, k + nx)]

        # skip masked voxel
        if distMask[idx] or np.alltrue(exTest[_volSlice].mask):
            it.iternext()
            continue

        _sqDD = ((exTest[_volSlice] - refimg) / refimg / dd) ** 2
        _gamma = np.sqrt(_sqDTA[idx] + _sqDD)

        # assign minimum values
        _mask = np.bitwise_and(gamma > _gamma, np.bitwise_not(_gamma.mask))
        gamma[_mask] = _gamma[_mask]

        it.iternext()

    return gamma


if __name__ == '__main__':
    def wave(x, y, z, v):
        return np.cos(v * x) + np.cos(v * y) + np.cos(v * z) \
            + 2 * (x ** 2 + y ** 2 + z ** 2)

    X, Y, Z = np.mgrid[-2:2:500j, -2:2:500j, -2:2:5j]
    img1 = wave(X, Y, Z, 10)
    img2 = wave(X, Y, Z, 12)

    gamma = gamma_index(img1, img2)
    print(gamma)

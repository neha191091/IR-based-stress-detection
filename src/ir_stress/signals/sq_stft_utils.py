"""Synchrosqueezing STFT utilities.

Adapted from the IR_iHR reference implementation:
  https://github.com/natalialmg/IR_iHR
  (upstream: IR_iHR/synchrosqueezing.py; based on Hau-Tieng Wu's SST code)

Original work: Martinez et al., ICIP 2019.

Included in this codebase but not yet evaluated in our proof of concept
(see docs/REPORT.md).
"""

from __future__ import annotations

import numpy as np
import scipy.fftpack
import scipy.special


def SST_STFT(
    x,
    lowFreq,
    highFreq,
    alpha,
    h=None,
    Dh=None,
    tDS=1,
    Smooth=True,
    Hemi=True,
):
    """Synchrosqueezed short-time Fourier transform.

    Ported from IR_iHR (github.com/natalialmg/IR_iHR).
    """
    xrow = x.size
    x = np.squeeze(x)

    t = np.arange(x.size)
    tLen = t[::tDS].size

    N = int(np.arange(-0.5 + alpha, 0.5, alpha).size + 1)
    Nrange = N // 2

    Lidx = int(np.round((N / 2) * (lowFreq / 0.5)))
    Hidx = int(np.round((N / 2) * (highFreq / 0.5)) - 1)
    fLen = int(Hidx - Lidx + 1)

    if highFreq > 0.5:
        raise ValueError("TopFreq must be a value in [0, 0.5]")
    if (tDS < 1) or (np.remainder(tDS, 1) != 0):
        raise ValueError("tDS must be an integer value >= 1")

    h = h.T
    hrow, hcol = h.shape
    h = np.squeeze(h)
    Dh = Dh.T
    Dh = np.squeeze(Dh)

    Lh = int((hrow - 1) / 2)
    if (hcol != 1) or (np.remainder(hrow, 2) == 0):
        raise ValueError("H must be a smoothing window with odd length")

    tfrtic = np.linspace(0, 0.5, N // 2)
    tfrsqtic = np.linspace(lowFreq, highFreq, fLen)
    tfrsq = np.zeros([tfrsqtic.size, tLen], dtype="complex")
    tfr = np.zeros([tfrtic.size, tLen], dtype="complex")

    Mid = int(np.round(tfrsqtic.size / 2))
    Delta = 20 * (tfrsqtic[2] - tfrsqtic[1]) ** 2
    weight = np.exp(-((tfrsqtic[Mid - 10 : Mid + 10] - tfrsqtic[Mid]) ** 2) / Delta)
    weight = weight / sum(weight)
    weightIDX = np.arange(Mid - 10, Mid + 10 + 1) - Mid

    for tidx in range(tLen):
        ti = int(t[(tidx - 1) * tDS + 1] + 1)
        A = int(-np.min(np.array([np.round(N / 2) - 1, Lh, ti - 1])))
        B = int(np.min(np.array([np.round(N / 2) - 1, Lh, xrow - ti])) + 1)
        ti = ti - 1
        tau = np.arange(A, B, dtype=int)
        indices = np.remainder(N + tau, N).astype(int)

        LhTau = (Lh + tau).astype(int)
        norm_h = np.linalg.norm(h[LhTau])

        tf0 = np.zeros(N)
        tf1 = np.zeros(N)
        tf0[indices] = x[ti + tau] * np.conj(h[LhTau]) / norm_h
        tf1[indices] = x[ti + tau] * np.conj(Dh[LhTau]) / norm_h

        tf0 = scipy.fftpack.fft(tf0)[:Nrange]
        tf1 = scipy.fftpack.fft(tf1)[:Nrange]

        omega = np.zeros(tf1.size)
        avoid_warn = np.where(tf0 != 0)
        omega[avoid_warn] = np.round(
            np.imag(N * tf1[avoid_warn] / tf0[avoid_warn] / (2.0 * np.pi))
        )

        sst = np.zeros(fLen, dtype="complex")

        for jcol in range(N // 2):
            jcolhat = jcol - omega[jcol]

            if (jcolhat <= Hidx) & (jcolhat >= Lidx):
                if Smooth:
                    IDXb = np.where(
                        (jcolhat - Lidx + weightIDX <= Hidx)
                        & (jcolhat - Lidx + weightIDX >= Lidx)
                    )
                    IDXa = (jcolhat - Lidx + weightIDX[IDXb]).astype(int)

                    if Hemi:
                        if np.real(tf0[jcol]) > 0:
                            sst[IDXa] = sst[IDXa] + tf0[jcol] * weight[IDXb]
                        else:
                            sst[IDXa] = sst[IDXa] - tf0[jcol] * weight[IDXb]
                    else:
                        sst[IDXa] = sst[IDXa] + tf0[jcol] * weight[IDXb]
                else:
                    idx = int(jcolhat - Lidx)
                    if Hemi:
                        if np.real(tf0[jcol]) > 0:
                            sst[idx] = sst[idx] + tf0[jcol]
                        else:
                            sst[idx] = sst[idx] - tf0[jcol]
                    else:
                        sst[idx] = sst[idx] + tf0[jcol]
        tfr[:, tidx] = tf0
        tfrsq[:, tidx] = sst

    return tfr, tfrtic, tfrsq, tfrsqtic


def hermf(N, M, tm):
    """Orthonormal Hermite functions and derivatives."""
    dt = 2 * tm / (N - 1)
    tt = np.linspace(-tm, tm, N)
    g = np.exp(-(tt**2) / 2)

    P = np.zeros([M + 1, N])
    Htemp = np.zeros([M + 1, N])
    Dh = np.zeros([M, N])

    P[0, :] = 1
    P[1, :] = 2 * tt
    for k in range(2, M):
        P[k, :] = 2 * tt * P[k - 1, :] - 2 * (k - 2) * P[k - 2, :]

    for k in range(M + 1):
        Htemp[k, :] = (
            P[k, :]
            * g
            / np.sqrt(np.sqrt(np.pi) * 2 ** (k + 1 - 1) * scipy.special.gamma(k + 1))
            * np.sqrt(dt)
        )
    h = Htemp[0:M, :]

    for k in range(M):
        Dh[k, :] = (tt * Htemp[k, :] - np.sqrt(2 * (k + 1)) * Htemp[k + 1, :]) * dt

    return h, Dh


def SST_helper(x, Hz, highFreq, lowFreq, windowLength=377):
    """Run synchrosqueezing on a 1D signal.

    Ported from IR_iHR (github.com/natalialmg/IR_iHR).
    """
    del Hz  # kept for API compatibility with IR_iHR
    frequency_axis_resolution = 0.001
    h, Dh = hermf(windowLength, 1, 2)
    return SST_STFT(
        x,
        lowFreq,
        highFreq,
        frequency_axis_resolution,
        h,
        Dh,
        1,
        False,
        False,
    )

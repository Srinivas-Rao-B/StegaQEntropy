import sys
import io
import os
import json
import time
import zlib
import socket
import base64
import random
import hashlib
import secrets
import textwrap
import subprocess
from pathlib import Path
import hmac

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity
from scipy.signal import convolve2d
from scipy.ndimage import label, sobel

try:
    import qiskit
    from qiskit import QuantumCircuit, transpile, qpy
    from qiskit_aer import AerSimulator
    HAVE_QISKIT = True
except Exception:
    HAVE_QISKIT = False

ML_COST_FACTOR = 0.75  # Soft priority factor for smooth ML pixels
SPREAD_TILE = 0
PE_FAST_SEARCH = False
META_BITS = 64

# Hard limits / protocol switches
MAX_CHANGED_PIXELS = 12000   # sender reports if exceeded
EMBED_SUBKEYS = False       # subkeys NOT embedded; QKD subkeys only seed the keystream
                             # (saves SUBKEY_BITS x chunks bits = fewer changed pixels)

# ML region map is hidden in the image as a fixed-size bitmap, in a RESERVED pixel set
# that the payload pass never touches -> every pixel changes by at most 1.
ML_GRID = 32             # ML regions = aligned GRID x GRID cells containing region centres
AUX_PIX_PER_BIT = 32     # reserved pixels per side-info bit
AUX_SALTS = 8            # fallback reserved-set variants (only used on rare overflow)
AUX_CFG = (100000, 0, 0.5, 0, 0)   # fixed (tau, first, fr, rnd, mode); no search

BASE_DIR = Path(__file__).resolve().parent

MESSAGE_JSON = BASE_DIR / "output" / "message_preparation" / "message_preparation.json"
CHUNKS_DIR = BASE_DIR / "output" / "message_preparation" / "chunks"
QKD_PACKAGE = BASE_DIR / "output" / "quantum_key_hierarchy" / "qkd_package.json"
QRNG_SEED = (
    BASE_DIR / "qrng" / "output" / "history_check" / "final_verified_seeds"
    / "adaptive_embedding_seed_verified.txt"
)
ADAPTIVE_REGION_MANIFEST = (
    BASE_DIR / "ml_module" / "output" / "surrogate_cnn_model"
    / "adaptive_embedding_region_manifest.json"
)
DWT_DIR = BASE_DIR / "output" / "dwt_decomposition"

SENDER_DIR = BASE_DIR / "output" / "adaptive_embedding" / "sender"
RECEIVER_DIR = BASE_DIR / "output" / "adaptive_embedding" / "receiver"
OUTPUT_DIR = SENDER_DIR  # sender-side helpers write here
STEGO_IMAGE = SENDER_DIR / "adaptive_stego_8bit.png"
PLAN_JSON = SENDER_DIR / "adaptive_embedding_plan.json"
METRICS_JSON = SENDER_DIR / "sender_metrics.json"
STEGO_VIEW_IMAGE = SENDER_DIR / "adaptive_stego_view.png"
EXTRACTED_MESSAGE = RECEIVER_DIR / "extracted_message.bin"
RECEIVER_METRICS_JSON = RECEIVER_DIR / "receiver_metrics.json"

HOST = "127.0.0.1"
PORT = 5055
WAVELET = "db2"

MASTER_KEY_BITS = 128
SUBKEY_BITS = 63
MASTER_QUBITS = 320
SUBKEY_QUBITS = 220
QKD_BATCH_SIZE = 32
QKD_MAX_RETRIES = 5
QBER_THRESHOLD = 0.11


def target_psnr(n_bits, shape):
    return float(10 * np.log10(255.0 ** 2 / ((n_bits / 2.0) / (shape[0] * shape[1]))))


def evaluate_structural_pattern_leakage(cover_u8, stego_u8, edge_threshold=0.15):
    diff_bin = (cover_u8 != stego_u8).astype(np.uint8)
    total_mods = int(np.sum(diff_bin))
    if total_mods == 0:
        return {
            "edge_alignment_ratio_pct": 0.0,
            "structural_contour_correlation": 0.0,
            "max_chain_component_size": 0,
            "mean_chain_component_size": 0.0,
            "morans_i": 0.0,
            "total_modifications": 0,
        }
    cover_f = cover_u8.astype(np.float64) / 255.0
    gx = sobel(cover_f, axis=1)
    gy = sobel(cover_f, axis=0)
    grad_mag = np.hypot(gx, gy)
    max_g = np.max(grad_mag)
    if max_g > 0:
        grad_mag /= max_g
    edge_mask = (grad_mag > edge_threshold).astype(np.uint8)
    ear = float(np.sum(diff_bin & edge_mask) / total_mods * 100.0)
    diff_f = diff_bin.astype(np.float64)
    c_grad = grad_mag - np.mean(grad_mag)
    c_diff = diff_f - np.mean(diff_f)
    denom = np.sqrt(np.sum(c_grad ** 2) * np.sum(c_diff ** 2))
    scc = float(np.sum(c_grad * c_diff) / (denom + 1e-12))
    labeled_array, num_features = label(diff_bin, structure=np.ones((3, 3), dtype=np.uint8))
    if num_features > 0:
        sizes = np.bincount(labeled_array.ravel())[1:]
        max_chain, mean_chain = int(np.max(sizes)), float(np.mean(sizes))
    else:
        max_chain, mean_chain = 0, 0.0
    w_kernel = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    z = diff_f - np.mean(diff_f)
    z_denom = np.sum(z ** 2)
    if z_denom < 1e-12:
        morans_i = 0.0
    else:
        z_lag = convolve2d(z, w_kernel, mode="same", boundary="symm")
        w_sum = np.sum(convolve2d(np.ones_like(diff_f), w_kernel, mode="same", boundary="symm"))
        morans_i = float((diff_bin.size / w_sum) * (np.sum(z * z_lag) / z_denom))
    return {
        "edge_alignment_ratio_pct": ear,
        "structural_contour_correlation": scc,
        "max_chain_component_size": max_chain,
        "mean_chain_component_size": mean_chain,
        "morans_i": morans_i,
        "total_modifications": total_mods,
    }


def _seed(key: bytes, label: bytes) -> int:
    return int.from_bytes(hashlib.sha512(key + label).digest(), "big")


def build_priority_mask(plan, shape, half=16):
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    for item in plan:
        r, c = int(item["row"]), int(item["col"])
        mask[max(0, r - half):min(h, r + half), max(0, c - half):min(w, c + half)] = True
    return mask


# ======================================================================
# ML REGION MAP: fixed-size bitmap hidden in a reserved pixel set
# ======================================================================
class _AuxFail(RuntimeError):
    pass


def _grid_dims(shape):
    return -(-shape[0] // ML_GRID), -(-shape[1] // ML_GRID)


def plan_to_bitmap(plan, shape):
    gh, gw = _grid_dims(shape)
    g = np.zeros((gh, gw), dtype=np.uint8)
    for it in plan:
        i = min(max(int(it["row"]) // ML_GRID, 0), gh - 1)
        j = min(max(int(it["col"]) // ML_GRID, 0), gw - 1)
        g[i, j] = 1
    return "".join(map(str, g.ravel()))


def mask_from_bitmap(bitmap, shape):
    gh, gw = _grid_dims(shape)
    g = np.array([int(b) for b in bitmap], dtype=np.uint8).reshape(gh, gw)
    big = np.kron(g, np.ones((ML_GRID, ML_GRID), dtype=np.uint8))
    return big[:shape[0], :shape[1]].astype(bool)


def side_nbits(shape):
    gh, gw = _grid_dims(shape)
    return gh * gw + 16            # bitmap + CRC16


def _crc16(bitmap):
    return zlib.crc32(bytes_from_bits(bitmap)) & 0xFFFF


def reserved_mask(key, shape, salt):
    n = shape[0] * shape[1]
    cnt = min(n, AUX_PIX_PER_BIT * side_nbits(shape))
    rng = np.random.default_rng(_seed(key, b"AUX-R" + str(salt).encode()))
    m = np.zeros(n, dtype=bool)
    m[rng.permutation(n)[:cnt]] = True
    return m.reshape(shape)


def _aux_embed(img_u8, bits_str, key, excluded_mask=None):
    tau, first, fr, rnd, mode = AUX_CFG
    c = np.asarray(img_u8, dtype=np.int64)
    rk = make_rank(key, c.shape)
    bits = [int(b) for b in bits_str]
    n1 = int(round(len(bits) * fr))
    wk = c.copy()
    r1 = _emb_spatial(wk, first, bits[:n1], tau, rnd, rk, mode, None, excluded_mask=excluded_mask)
    r2 = None if r1 is None else _emb_spatial(wk, 1 - first, bits[n1:], tau, rnd, rk, mode, None,
                                              excluded_mask=excluded_mask)
    if r1 is None or r2 is None or wk.min() < 0 or wk.max() > 255:
        raise _AuxFail("ML side-info aux embedding failed.")
    return wk.astype(np.uint8)


def _aux_extract(img_u8, nbits, key, excluded_mask=None):
    tau, first, fr, rnd, mode = AUX_CFG
    sg = np.asarray(img_u8, dtype=np.int64)
    rk = make_rank(key, sg.shape)
    n1 = int(round(nbits * fr))
    wk = sg.copy()
    b2 = _ext_spatial(wk, 1 - first, nbits - n1, tau, rnd, rk, mode, None, excluded_mask)
    b1 = None if b2 is None else _ext_spatial(wk, first, n1, tau, rnd, rk, mode, None, excluded_mask)
    if b1 is None or len(b1) != n1 or len(b2) != nbits - n1:
        raise _AuxFail("ML side-info aux extraction failed.")
    return "".join(map(str, list(b1) + list(b2))), wk.astype(np.uint8)


def embed_with_side_info(cover_u8, stream, key, bitmap, ml):
    """Stage B (payload, reserved pixels excluded) then stage A (bitmap, reserved pixels only)."""
    shape = cover_u8.shape
    nb = side_nbits(shape)
    plain = bitmap + f"{_crc16(bitmap):016b}"
    enc = xor_str(plain, keystream(key, b"|MLMAP|", nb))
    last = None
    for salt in range(AUX_SALTS):
        R = reserved_mask(key, shape, salt)
        try:
            stego_b, info = pe_embed(cover_u8, stream, key, ml, R)
            final = _aux_embed(stego_b, enc, key, ~R)
            info["salt"] = salt
            return final, info
        except _AuxFail as e:
            last = e
    raise RuntimeError(f"Side-info embedding failed for all salts: {last}")


def extract_side_info(final_u8, key):
    """Undo stage A first (needs no ML mask). Returns (bitmap, stego_B, reserved mask)."""
    shape = final_u8.shape
    nb = side_nbits(shape)
    for salt in range(AUX_SALTS):
        R = reserved_mask(key, shape, salt)
        try:
            enc, img_b = _aux_extract(final_u8, nb, key, ~R)
        except _AuxFail:
            continue
        plain = xor_str(enc, keystream(key, b"|MLMAP|", nb))
        bitmap, chk = plain[:-16], int(plain[-16:], 2)
        if _crc16(bitmap) == chk:
            return bitmap, img_b, R
    raise RuntimeError("ML side-info not found (wrong key or corrupted stego).")


# ----------------------------------------------------------------------
TAUS = (1200, 100000)
FRS = [round(0.46 + 0.01 * i, 2) for i in range(9)]  # 0.46 .. 0.54
CFGS = [
    (tau, f, fr, rnd, mode)
    for mode in (0,)
    for rnd in (3, 2, 0)
    for f in (0, 1)
    for tau in TAUS
    for fr in FRS
]

_G = {}


def make_rank(key, shape):
    rng = np.random.default_rng(_seed(key, b"PE-RANK"))
    return rng.permutation(shape[0] * shape[1]).reshape(shape)


# ======================================================================
# 1) GEOMETRY, 12-NEIGHBOUR ROUGHNESS KEY, PREDICTOR + ML INTEGRATION
# ======================================================================
def _geom(shape, layer):
    k = (shape, layer)
    if k not in _G:
        h, w = shape
        rr, cc = np.mgrid[1:h - 1, 1:w - 1]
        m = ((rr + cc) & 1) == layer
        _G[k] = (rr[m], cc[m])
    return _G[k]


def _order_spatial(img, layer, rnd, rank, mode, ml_mask=None, excluded_mask=None):
    rr, cc = _geom(img.shape, layer)

    # Reflect padding keeps all ring samples strictly on the opposite parity layer
    p = np.pad(img.astype(np.int64), 2, mode="reflect")
    R, C = rr + 2, cc + 2
    n = p[R - 1, C]
    s = p[R + 1, C]
    w = p[R, C - 1]
    e = p[R, C + 1]
    ring = np.stack([
        p[R - 1, C - 2], p[R - 1, C + 2], p[R + 1, C - 2], p[R + 1, C + 2],
        p[R - 2, C - 1], p[R - 2, C + 1], p[R + 2, C - 1], p[R + 2, C + 1],
    ])

    dh = np.abs(w - e)
    dv = np.abs(n - s)

    wh = 1.0 / (dh.astype(np.float64) + 1.0)
    wv = 1.0 / (dv.astype(np.float64) + 1.0)
    pred_f = (wh * (w + e) + wv * (n + s)) / (2.0 * (wh + wv))
    if mode == 1:
        pred_f = 0.8 * pred_f + 0.2 * (ring.sum(0) / 8.0)
    pred = np.floor(pred_f + (rnd * 0.25)).astype(np.int64)

    # 12-neighbor deterministic roughness
    S12 = n + s + w + e + ring.sum(0)
    base_cx = dh + dv + np.abs(n + s - w - e)
    ring_dev = np.abs(ring * 12 - S12).sum(0)
    cx = (12 * base_cx + ring_dev // 4).astype(np.float64)

    # ML GUIDANCE:
    # Prioritize pixels inside ML regions if their local roughness is smooth-to-moderate.
    # This prevents non-carrier shift cascades on noisy edges inside ML boxes.
    if ml_mask is not None and ML_COST_FACTOR != 1.0:
        inml = ml_mask[rr, cc]
        is_candidate_ml = inml & (base_cx <= 4)
        cx = np.where(is_candidate_ml, cx * ML_COST_FACTOR, cx)

    if excluded_mask is not None:
        cx = np.where(excluded_mask[rr, cc], 1.0e12, cx)   # never active (tau < 1e12)

    cx_int = cx.astype(np.int64)
    o = np.lexsort((rank[rr, cc], cx_int))
    return rr[o], cc[o], pred[o], cx_int[o]


# ======================================================================
# 2) GUARD-FREE REVERSIBLE PEE (Double-Bin Expansion)
# ======================================================================
def _emb_spatial(img, layer, bits, tau, rnd, rank, mode, ml_mask=None, cache=None, ckey=None,
                 excluded_mask=None):
    m = len(bits)
    if m == 0:
        return 0

    if cache is not None and ckey in cache:
        rr, cc, pred, cx = cache[ckey]
    else:
        rr, cc, pred, cx = _order_spatial(img, layer, rnd, rank, mode, ml_mask, excluded_mask)
        if cache is not None:
            cache[ckey] = (rr, cc, pred, cx)

    x = img[rr, cc]
    e = x - pred

    safe = (pred >= 1) & (pred <= 254)
    active = (cx <= tau) & safe

    z = ((e == 0) | (e == -1)) & active
    cum_z = np.cumsum(z)

    if len(cum_z) == 0 or cum_z[-1] < m:
        return None

    cutoff = int(np.searchsorted(cum_z, m)) + 1
    xa = x[:cutoff]
    ea = e[:cutoff]
    za = z[:cutoff]
    act_a = active[:cutoff]

    carrier_pos = np.where(za)[0][:m]
    b = np.asarray(bits, dtype=np.int64)

    add = np.where(act_a, np.where(ea > 0, 1, np.where(ea < -1, -1, 0)), 0)
    add[carrier_pos] = np.where(ea[carrier_pos] == 0, b, -b)

    img[rr[:cutoff], cc[:cutoff]] = xa + add
    return int(np.count_nonzero(add))


def _ext_spatial(img, layer, m, tau, rnd, rank, mode, ml_mask=None, excluded_mask=None):
    if m == 0:
        return np.zeros(0, dtype=np.uint8)

    rr, cc, pred, cx = _order_spatial(img, layer, rnd, rank, mode, ml_mask, excluded_mask)
    x = img[rr, cc]
    e = x - pred

    safe = (pred >= 1) & (pred <= 254)
    active = (cx <= tau) & safe

    carr = ((e >= -2) & (e <= 1)) & active
    cum_c = np.cumsum(carr)

    if len(cum_c) == 0 or cum_c[-1] < m:
        return None

    cutoff = int(np.searchsorted(cum_c, m)) + 1
    xa = x[:cutoff]
    ea = e[:cutoff]
    ca = carr[:cutoff]
    act_a = active[:cutoff]

    carrier_pos = np.where(ca)[0][:m]
    ec = ea[carrier_pos]

    out = ((ec == 1) | (ec == -2)).astype(np.uint8)

    orig = np.where(act_a, np.where(ea > 1, xa - 1, np.where(ea < -2, xa + 1, xa)), xa)
    orig[carrier_pos] = np.where(
        ec == 1,
        xa[carrier_pos] - 1,
        np.where(ec == -2, xa[carrier_pos] + 1, xa[carrier_pos])
    )

    img[rr[:cutoff], cc[:cutoff]] = orig
    return out


# ======================================================================
# 3) PARAMETER SEARCH & OPTIMIZATION (Synchronized with ml_mask)
# ======================================================================
def pe_embed(cover_u8, bits_str, key, ml_mask, excluded_mask=None):
    c = np.asarray(cover_u8, dtype=np.int64)
    bits = [int(b) for b in bits_str]
    n = len(bits)
    rk = make_rank(key, c.shape)
    best = None
    cache = {}

    for cfg in CFGS:
        tau, first, fr, rnd, mode = cfg
        n1 = int(round(n * fr))
        wk = c.copy()

        r1 = _emb_spatial(wk, first, bits[:n1], tau, rnd, rk, mode,
                          ml_mask, cache, (first, rnd, mode), excluded_mask)
        if r1 is None:
            continue
        r2 = _emb_spatial(wk, 1 - first, bits[n1:], tau, rnd, rk, mode,
                          ml_mask, excluded_mask=excluded_mask)
        if r2 is None:
            continue

        if wk.min() < 0 or wk.max() > 255:
            continue

        ch = int(np.count_nonzero(wk != c))
        mse = float(np.mean((wk - c) ** 2))

        if best is None or ch < best[0] or (ch == best[0] and mse < best[1]):
            best = (ch, mse, wk, cfg)

    if best is None:
        raise RuntimeError("Payload does not fit with any configuration.")

    ch, mse, st_arr, best_cfg = best
    st = st_arr.astype(np.uint8)
    return st, {
        "changes": ch,
        "config": best_cfg,
        "nbits": n,
        "bits_per_change": n / max(ch, 1),
    }


def pe_extract(stego_u8, nbits, key, ml_mask, verify, excluded_mask=None):
    sg = np.asarray(stego_u8, dtype=np.int64)
    rk = make_rank(key, sg.shape)

    for cfg in CFGS:
        tau, first, fr, rnd, mode = cfg
        n1 = int(round(nbits * fr))
        wk = sg.copy()

        b2 = _ext_spatial(wk, 1 - first, nbits - n1, tau, rnd, rk, mode, ml_mask, excluded_mask)
        b1 = None if b2 is None else _ext_spatial(wk, first, n1, tau, rnd, rk, mode, ml_mask, excluded_mask)
        if b1 is None or len(b1) != n1 or len(b2) != (nbits - n1):
            continue

        out = "".join(map(str, list(b1) + list(b2)))
        if verify(out):
            return out, wk.astype(np.uint8), cfg

    raise RuntimeError("Extraction failed: no configuration verified.")




def calculate_fsim(org_img, mod_img):
    im1 = np.asarray(org_img, dtype=np.float64)
    im2 = np.asarray(mod_img, dtype=np.float64)
    if im1.shape != im2.shape:
        raise ValueError("Images must have identical dimensions for FSIM.")
    rows, cols = im1.shape
    dx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float64) / 16.0
    dy = dx.T
    gm1 = np.hypot(convolve2d(im1, dx, mode="same", boundary="symm"),
                   convolve2d(im1, dy, mode="same", boundary="symm"))
    gm2 = np.hypot(convolve2d(im2, dx, mode="same", boundary="symm"),
                   convolve2d(im2, dy, mode="same", boundary="symm"))
    nscale, norient, min_wave, mult, sigma_on_f = 4, 4, 3, 2.0, 0.55
    y, x = np.mgrid[-rows // 2:int(np.ceil(rows / 2)), -cols // 2:int(np.ceil(cols / 2))]
    radius = np.hypot(x / max(cols, 1), y / max(rows, 1))
    radius[rows // 2, cols // 2] = 1.0
    theta = np.arctan2(-y, x)
    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)
    filters = []
    for s in range(nscale):
        fo = 1.0 / (min_wave * (mult ** s))
        log_gabor = np.exp(-((np.log(radius / fo)) ** 2) / (2 * (np.log(sigma_on_f) ** 2)))
        log_gabor[radius > 0.5] = 0.0
        for o in range(norient):
            angle = o * np.pi / norient
            ds = np.sin(theta) * np.cos(angle) - np.cos(theta) * np.sin(angle)
            dc = np.cos(theta) * np.cos(angle) + np.sin(theta) * np.sin(angle)
            dtheta = np.abs(np.arctan2(ds, dc))
            filters.append(log_gabor * np.exp(-(dtheta ** 2) / (2 * (0.35 ** 2))))
    fft1, fft2 = np.fft.fft2(im1), np.fft.fft2(im2)
    n1 = np.zeros((rows, cols))
    n2 = np.zeros((rows, cols))
    d1 = np.zeros((rows, cols))
    d2 = np.zeros((rows, cols))
    for filt in filters:
        eo1, eo2 = np.fft.ifft2(fft1 * filt), np.fft.ifft2(fft2 * filt)
        n1 += np.abs(eo1)
        n2 += np.abs(eo2)
        d1 += np.hypot(eo1.real, eo1.imag)
        d2 += np.hypot(eo2.real, eo2.imag)
    pc1, pc2 = n1 / (d1 + 1e-4), n2 / (d2 + 1e-4)
    T1, T2 = 0.85, 160.0
    s_pc = (2.0 * pc1 * pc2 + T1) / (pc1 ** 2 + pc2 ** 2 + T1)
    s_gm = (2.0 * gm1 * gm2 + T2) / (gm1 ** 2 + gm2 ** 2 + T2)
    pc_max = np.maximum(pc1, pc2)
    return float(np.clip(np.sum(s_pc * s_gm * pc_max) / (np.sum(pc_max) + 1e-12), 0.0, 1.0))


def metrics_8bit(cover_u8, stego_u8):
    a = np.asarray(cover_u8, dtype=np.float64)
    b = np.asarray(stego_u8, dtype=np.float64)
    diff = b - a
    mse = float(np.mean(diff ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
    return {
        "psnr": psnr,
        "mse": mse,
        "mae": float(np.mean(np.abs(diff))),
        "ssim": float(structural_similarity(a, b, data_range=255.0)),
        "fsim": float(calculate_fsim(a, b)),
        "npcr": float(np.mean(diff != 0) * 100),
        "uaci": float(np.mean(np.abs(diff) / 255.0) * 100),
        "max_change": float(np.max(np.abs(diff))),
    }


class QRNG:
    def __init__(self, path, role="alice"):
        text = path.read_text(encoding="utf-8")
        allbits = self._extract(text)
        half = len(allbits) // 2
        self.bits = allbits[:half] if role == "alice" else allbits[half:]
        self.pointer = 0

    def _extract(self, text):
        result, active = [], False
        for line in text.splitlines():
            value = line.strip()
            if value.lower() == "binary":
                active = True
                continue
            if active and value.lower().startswith("hexadecimal"):
                break
            if active and value and set(value).issubset({"0", "1"}):
                result.append(value)
        bits = "".join(result)
        if not bits:
            raise ValueError("QRNG binary section not found.")
        return bits

    def take(self, count):
        n = len(self.bits)
        if n == 0:
            raise RuntimeError("QRNG has no bits.")
        out, need = [], count
        while need > 0:
            if self.pointer >= n:      # exhausted -> wrap to the start
                self.pointer = 0
            part = self.bits[self.pointer:self.pointer + need]
            out.append(part)
            self.pointer += len(part)
            need -= len(part)
        return "".join(out)

    def remaining(self):
        return len(self.bits) - self.pointer


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def send_packet(sock, payload):
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock.sendall(len(raw).to_bytes(8, "big"))
    sock.sendall(raw)


def receive_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        block = sock.recv(size - len(data))
        if not block:
            raise ConnectionError("TCP connection closed.")
        data.extend(block)
    return bytes(data)


def receive_packet(sock):
    size = int.from_bytes(receive_exact(sock, 8), "big")
    return json.loads(receive_exact(sock, size).decode("utf-8"))


def load_chunks():
    files = list(CHUNKS_DIR.glob("*.bin"))
    if not files:
        raise FileNotFoundError(f"No .bin chunks found in {CHUNKS_DIR}")

    def chunk_number(path):
        digits = "".join(c for c in path.stem if c.isdigit())
        return int(digits) if digits else 10 ** 9

    files.sort(key=chunk_number)
    chunks = {}
    for path in files:
        raw = path.read_bytes()
        try:
            text = raw.decode("ascii")
        except UnicodeDecodeError:
            text = None
        if text is not None and text and all(bit in "01" for bit in text):
            bits = text.strip()
            if not bits:
                raise RuntimeError(f"Chunk {path.stem} is empty.")
        else:
            bits = "".join(f"{byte:08b}" for byte in raw)
        chunks[path.stem] = {
            "chunk_id": path.stem,
            "bytes": raw,
            "bits": bits,
            "bit_length": len(bits),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return chunks


def load_message_metadata():
    return load_json(MESSAGE_JSON)


def load_qkd_blueprint():
    return load_json(QKD_PACKAGE)


def normalize_region_id(value):
    if isinstance(value, int):
        return value
    text = str(value)
    if text.isdigit():
        return int(text)
    if "_" in text:
        tail = text.split("_")[-1]
        if tail.isdigit():
            return int(tail)
    return text


def load_authoritative_plan(required=True):
    if not ADAPTIVE_REGION_MANIFEST.exists():
        if required:
            raise FileNotFoundError(ADAPTIVE_REGION_MANIFEST)
        print("[WARN] ML region manifest not found -> no ML priority.")
        return []
    manifest = load_json(ADAPTIVE_REGION_MANIFEST)
    regions = manifest.get("regions")
    if not isinstance(regions, list):
        raise ValueError("Adaptive embedding manifest must contain a 'regions' list.")
    plan = []
    for record in regions:
        for field in ("region_id", "row", "col"):
            if field not in record:
                raise ValueError(f"Region {record.get('region_id')} is missing '{field}'.")
        plan.append({
            "region_id": normalize_region_id(record["region_id"]),
            "row": int(record["row"]),
            "col": int(record["col"]),
            "payload_bits": int(record.get("payload_bits", 0)),
        })
    return plan


def load_dwt():
    result = {}
    for band in ["LL", "LH", "HL", "HH"]:
        path = DWT_DIR / f"{band}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        result[band] = np.load(path).astype(np.float64)
    return result


def reconstruct(dwt):
    return pywt.idwt2((dwt["LL"], (dwt["LH"], dwt["HL"], dwt["HH"])), WAVELET, mode="periodization")


def build_cover_u8():
    return np.rint(np.clip(reconstruct(load_dwt()), 0.0, 255.0)).astype(np.uint8)


def bytes_from_bits(bits):
    if len(bits) % 8:
        bits = bits + "0" * (8 - len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def bits_from_bytes(data):
    return "".join(f"{byte:08b}" for byte in data)


def xor_str(a, b):
    n = len(a)
    if n == 0:
        return ""
    return format(int(a, 2) ^ int(b[:n], 2), f"0{n}b")


def hash_expand(seed, bit_count):
    out = bytearray()
    counter = 0
    required = (bit_count + 7) // 8
    while len(out) < required:
        out.extend(hashlib.sha256(seed + counter.to_bytes(8, "big")).digest())
        counter += 1
    return bytes(out[:required])


def keystream(key, label, nbits):
    return bits_from_bytes(hash_expand(key + label, nbits))[:nbits]


def derive_key(bits, label, size):
    raw = bytes_from_bits(bits)
    material = hashlib.sha256(raw + label.encode("utf-8")).digest()
    return hash_expand(material, size)


def total_stream_bits(chunk_lens):
    return META_BITS + sum((SUBKEY_BITS if EMBED_SUBKEYS else 0) + L for L in chunk_lens)


def build_stream(chunks, chunk_lens, chunk_ids, subkeys, master_key):
    message_bits = "".join(chunks[cid]["bits"] for cid in chunk_ids)
    header = f"{len(message_bits):032b}" + f"{zlib.crc32(bytes_from_bits(message_bits)) & 0xffffffff:032b}"
    parts = [header]
    for cid, L in zip(chunk_ids, chunk_lens):
        body = chunks[cid]["bits"]
        sk = subkeys[cid]
        ks = keystream(sk, b"|BODY|", L)
        parts.append((bits_from_bytes(sk)[:SUBKEY_BITS] if EMBED_SUBKEYS else "") + xor_str(body, ks))
    plain = "".join(parts)
    return xor_str(plain, keystream(master_key, b"|STREAM|", len(plain)))


def make_verifier(chunk_lens, chunk_ids, subkeys, master_key):
    n = total_stream_bits(chunk_lens)
    msg_bits = sum(chunk_lens)

    def verify(stream):
        plain = xor_str(stream, keystream(master_key, b"|STREAM|", n))
        if int(plain[:32], 2) != msg_bits:
            return False
        if not EMBED_SUBKEYS:
            return parse_stream(stream, chunk_lens, chunk_ids, subkeys, master_key)[1]
        pos = META_BITS
        for cid, L in zip(chunk_ids, chunk_lens):
            if plain[pos:pos + SUBKEY_BITS] != bits_from_bytes(subkeys[cid])[:SUBKEY_BITS]:
                return False
            pos += SUBKEY_BITS + L
        return True
    return verify


def parse_stream(stream, chunk_lens, chunk_ids, subkeys, master_key):
    plain = xor_str(stream, keystream(master_key, b"|STREAM|", len(stream)))
    crc_expected = int(plain[32:64], 2)
    pos, out = META_BITS, []
    for cid, L in zip(chunk_ids, chunk_lens):
        pos += SUBKEY_BITS if EMBED_SUBKEYS else 0
        ks = keystream(subkeys[cid], b"|BODY|", L)
        out.append(xor_str(plain[pos:pos + L], ks))
        pos += L
    message_bits = "".join(out)
    crc_ok = (zlib.crc32(bytes_from_bits(message_bits)) & 0xffffffff) == crc_expected
    return message_bits, crc_ok


def make_bb84_circuit(bits, bases):
    circuit = QuantumCircuit(len(bits), len(bits))
    for index, bit in enumerate(bits):
        if bit:
            circuit.x(index)
        if bases[index]:
            circuit.h(index)
    return circuit


def measure_bb84_batch(circuits, bases_list):
    measured_circuits = []
    for circuit, bases in zip(circuits, bases_list):
        measured = circuit.copy()
        for index, basis in enumerate(bases):
            if basis:
                measured.h(index)
        num_q = measured.num_qubits
        if not measured.cregs:
            measured.add_register(qiskit.circuit.ClassicalRegister(num_q))
        measured.measure(range(num_q), range(num_q))
        measured_circuits.append(measured)
    backend = AerSimulator(method="stabilizer")
    compiled = transpile(measured_circuits, backend, optimization_level=0)
    result = backend.run(compiled, shots=1).result()
    all_bits = []
    for index in range(len(measured_circuits)):
        counts = result.get_counts(index)
        bitstring = next(iter(counts)).replace(" ", "")
        all_bits.extend(int(bit) for bit in bitstring[::-1])
    return all_bits

def qkd_alice(sock, qrng, session_id, required_bits, raw_qubits, sample_ratio=0.1):
    alice_bits = [int(b) for b in qrng.take(raw_qubits)]
    alice_bases = [int(b) for b in qrng.take(raw_qubits)]

    # 1. Simulate the quantum transmission (In simulation: pass serialized circuits as opaque qubits)
    serialized = []
    for start in range(0, raw_qubits, QKD_BATCH_SIZE):
        end = min(start + QKD_BATCH_SIZE, raw_qubits)
        circuit = make_bb84_circuit(alice_bits[start:end], alice_bases[start:end])
        buffer = io.BytesIO()
        qpy.dump(circuit, buffer)
        serialized.append(base64.b64encode(buffer.getvalue()).decode("ascii"))

    send_packet(sock, {
        "type": "qkd_quantum_transmit",
        "session_id": session_id,
        "raw_qubits": raw_qubits,
        "batch_size": QKD_BATCH_SIZE,
        "circuits": serialized,
    })

    # 2. Receive ONLY Bob's measurement bases over the classical socket (NO BITS)
    bob_response = receive_packet(sock)
    if bob_response["type"] != "qkd_bob_bases" or bob_response["session_id"] != session_id:
        raise RuntimeError("Invalid basis reconciliation response from Bob.")
    bob_bases = [int(b) for b in bob_response["bob_bases"]]

    # 3. Basis reconciliation: Alice sends matching basis indices to Bob
    match_indices = [i for i in range(raw_qubits) if alice_bases[i] == bob_bases[i]]
    send_packet(sock, {
        "type": "qkd_matching_indices",
        "session_id": session_id,
        "matching_indices": match_indices,
    })

    if len(match_indices) < required_bits:
        raise RuntimeError(f"Insufficient sifted bits in {session_id}: {len(match_indices)} < {required_bits}")

    # 4. Sacrificial QBER estimation: sample a subset, reveal them, and DISCARD them
    num_sample = max(1, int(len(match_indices) * sample_ratio))
    rng = random.Random(_seed(session_id.encode(), b"SAMPLE"))
    sample_sub_indices = sorted(rng.sample(range(len(match_indices)), num_sample))

    sample_positions = [match_indices[idx] for idx in sample_sub_indices]
    alice_sample_bits = [alice_bits[pos] for pos in sample_positions]

    send_packet(sock, {
        "type": "qkd_qber_sample",
        "session_id": session_id,
        "sample_indices": sample_sub_indices,
        "sample_bits": alice_sample_bits,
    })

    qber_packet = receive_packet(sock)
    if qber_packet["type"] != "qkd_qber_result" or qber_packet["session_id"] != session_id:
        raise RuntimeError("Expected QBER estimation from Bob.")
    qber = float(qber_packet["qber"])

    if qber > QBER_THRESHOLD:
        raise RuntimeError(f"QBER too high for {session_id}: {qber:.4f}")

    # 5. Form the final raw key using ONLY the unrevealed, non-sacrificed sifted bits
    sample_set = set(sample_sub_indices)
    unrevealed_positions = [match_indices[i] for i in range(len(match_indices)) if i not in sample_set]
    sifted_alice = [alice_bits[pos] for pos in unrevealed_positions]

    if len(sifted_alice) < required_bits:
        raise RuntimeError(f"Not enough secret bits after sacrificial QBER discard: {len(sifted_alice)} < {required_bits}")

    key = derive_key("".join(map(str, sifted_alice)), session_id, required_bits)
    
    # 6. Authenticate without sending the full key hash (send short auth tag)
    tag = hmac.new(key, b"QKD_AUTH_VERIFY", hashlib.sha256).hexdigest()[:16]
    send_packet(sock, {"type": "qkd_auth_tag", "session_id": session_id, "tag": tag})

    confirmation = receive_packet(sock)
    if confirmation.get("status") != "OK":
        raise RuntimeError("QKD key authentication tag mismatch.")

    return {"key": key, "qber": qber, "sifted_bits": len(sifted_alice)}


def qkd_bob(sock, qrng, required_bits):
    # 1. Receive simulated quantum payload
    packet = receive_packet(sock)
    if packet["type"] != "qkd_quantum_transmit":
        raise RuntimeError("Expected quantum circuit transmission.")
    session_id = packet["session_id"]
    raw_qubits = int(packet["raw_qubits"])
    print(f"QKD {session_id}: measuring {raw_qubits} qubits", flush=True)
    circuits = [qpy.load(io.BytesIO(base64.b64decode(enc)))[0] for enc in packet["circuits"]]
    bob_bases = [int(b) for b in qrng.take(raw_qubits)]

    # Measure batch
    bases_list, pointer = [], 0
    for circuit in circuits:
        count = circuit.num_qubits
        bases_list.append(bob_bases[pointer:pointer + count])
        pointer += count

    bob_bits = measure_bb84_batch(circuits, bases_list)
    print(f"QKD {session_id}: measurement complete", flush=True)

    # 2. Send ONLY bases over the wire (NO BITS)
    send_packet(sock, {
        "type": "qkd_bob_bases",
        "session_id": session_id,
        "bob_bases": bob_bases,
    })

    # 3. Receive indices where bases matched
    reconcile = receive_packet(sock)
    if reconcile["type"] != "qkd_matching_indices" or reconcile["session_id"] != session_id:
        raise RuntimeError("Expected basis match indices.")
    match_indices = reconcile["matching_indices"]

    # 4. Receive sacrificial sample from Alice and compute QBER
    sample_pkt = receive_packet(sock)
    if sample_pkt["type"] != "qkd_qber_sample" or sample_pkt["session_id"] != session_id:
        raise RuntimeError("Expected sacrificial sample.")

    sample_sub_indices = sample_pkt["sample_indices"]
    alice_sample_bits = sample_pkt["sample_bits"]

    sample_positions = [match_indices[idx] for idx in sample_sub_indices]
    bob_sample_bits = [bob_bits[pos] for pos in sample_positions]

    errors = sum(a != b for a, b in zip(alice_sample_bits, bob_sample_bits))
    qber = errors / len(alice_sample_bits) if alice_sample_bits else 0.0

    send_packet(sock, {
        "type": "qkd_qber_result",
        "session_id": session_id,
        "qber": qber,
    })

    if qber > QBER_THRESHOLD:
        raise RuntimeError(f"QBER exceeds threshold: {qber:.4f}")

    # 5. Extract unrevealed bits (discard the sample)
    sample_set = set(sample_sub_indices)
    unrevealed_positions = [match_indices[i] for i in range(len(match_indices)) if i not in sample_set]
    sifted_bob = [bob_bits[pos] for pos in unrevealed_positions]

    key = derive_key("".join(map(str, sifted_bob)), session_id, required_bits)

    # 6. Check short HMAC auth tag
    auth_pkt = receive_packet(sock)
    tag_expected = hmac.new(key, b"QKD_AUTH_VERIFY", hashlib.sha256).hexdigest()[:16]
    if not secrets.compare_digest(auth_pkt["tag"], tag_expected):
        send_packet(sock, {"type": "qkd_auth_status", "status": "FAIL"})
        raise RuntimeError("QKD key authentication failed.")

    send_packet(sock, {"type": "qkd_auth_status", "status": "OK"})
    return {
        "session_id": session_id,
        "key": key,
        "qber": qber,
        "sifted_bits": len(sifted_bob),
    }
_EPS = 1e-12


def _prob_hist(arr, hi=256.0):
    h = np.histogram(arr, bins=256, range=(0.0, hi), density=True)[0] + _EPS
    return h / np.sum(h)


def _entropy(arr):
    h = _prob_hist(arr)
    return float(-np.sum(h * np.log2(h)))


def _row(label, value):
    print(f"{label:<30}: {value}")


def save_selected_regions_plot(operations):
    if not operations:
        return None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    xs = [float(o["col"]) for o in operations]
    ys = [float(o["row"]) for o in operations]
    ids = [str(o.get("region_id", "?")) for o in operations]
    plot_path = OUTPUT_DIR / "selected_embedding_regions.png"
    plt.figure(figsize=(12, 10))
    plt.scatter(xs, ys, s=35, marker="o", label="ML Priority Region")
    for x, y, rid in zip(xs, ys, ids):
        plt.annotate(rid, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6)
    plt.xlabel("X Coordinate (Column)")
    plt.ylabel("Y Coordinate (Row)")
    plt.title("ML Priority Regions for Adaptive Pixel Embedding")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Region plot saved: {plot_path}")
    return plot_path


def calculate_srm_features(image):
    import sealwatch as sw
    image = np.asarray(image, dtype=np.uint8)
    grouped = sw.srm.extract(image)
    feats = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in grouped.values()])
    if not np.all(np.isfinite(feats)):
        raise ValueError("SRM produced non-finite feature values.")
    return feats


def calculate_spam_features(image):
    import sealwatch as sw
    image = np.asarray(image, dtype=np.uint8)
    f = sw.spam.extract(image)
    if isinstance(f, dict):
        f = np.concatenate([np.asarray(v).ravel() for v in f.values()])
    else:
        f = np.asarray(f).ravel()
    return f.astype(np.float64)


def _feature_diff(fn, name, cover_u8, stego_u8):
    try:
        a, b = fn(cover_u8), fn(stego_u8)
        d = b - a
        print(f"\n{name} Analysis:")
        print(f"    Feature dimensions       : {a.size}")
        print(f"    Feature difference L2    : {np.linalg.norm(d):.6f}")
        print(f"    Feature difference MAE   : {np.mean(np.abs(d)):.6f}")
        print(f"    Feature difference max   : {np.max(np.abs(d)):.6f}")
        return {
            "feature_dimensions": int(a.size),
            "difference_l2": float(np.linalg.norm(d)),
            "difference_mae": float(np.mean(np.abs(d))),
            "difference_max": float(np.max(np.abs(d))),
        }
    except Exception as exc:
        print(f"\n{name} skipped ({exc}).")
        return {
            "feature_dimensions": 0,
            "difference_l2": float("nan"),
            "difference_mae": float("nan"),
            "difference_max": float("nan"),
        }


def calculate_and_print_steganalysis(cover_image, stego_image):
    cover_u8 = np.rint(np.clip(cover_image, 0.0, 255.0)).astype(np.uint8)
    stego_u8 = np.rint(np.clip(stego_image, 0.0, 255.0)).astype(np.uint8)

    def flip_negative(pixel):
        pix = pixel.astype(np.int16)
        res = np.where(pix % 2 == 0, pix - 1, pix + 1)
        return np.clip(res, 0, 255).astype(np.uint8)

    def analyze_rs(image, mask=(0, 1, 1, 0)):
        h, w = image.shape
        g = len(mask)
        n = (h * w) // g
        groups = image.flatten()[:n * g].reshape(n, g)
        pos, neg = groups.copy(), groups.copy()
        for idx, m in enumerate(mask):
            if m == 1:
                pos[:, idx] = groups[:, idx] ^ 1
                neg[:, idx] = flip_negative(groups[:, idx])
            elif m == -1:
                pos[:, idx] = flip_negative(groups[:, idx])
                neg[:, idx] = groups[:, idx] ^ 1

        def var(a):
            return np.sum(np.abs(np.diff(a.astype(np.float64), axis=1)), axis=1)

        v0, vp, vn = var(groups), var(pos), var(neg)
        r_m, s_m = np.mean(vp > v0) * 100.0, np.mean(vp < v0) * 100.0
        r_n, s_n = np.mean(vn > v0) * 100.0, np.mean(vn < v0) * 100.0
        d0, d1 = r_m - s_m, r_n - s_n
        denom = d0 + d1
        p_est = abs(d0 - d1) / (abs(denom) + 1e-8) if abs(denom) > 1e-6 else 0.0
        return {
            "r_m": float(r_m), "s_m": float(s_m), "r_neg_m": float(r_n), "s_neg_m": float(s_n),
            "diff_r": float(abs(r_m - r_n)), "diff_s": float(abs(s_m - s_n)),
            "estimated_payload_p": float(np.clip(p_est, 0.0, 1.0) * 100.0),
        }

    cover_rs, stego_rs = analyze_rs(cover_u8), analyze_rs(stego_u8)
    rs_diff_p = float(stego_rs["estimated_payload_p"] - cover_rs["estimated_payload_p"])
    r1 = np.diff(cover_u8.astype(np.float64), axis=1)
    r2 = np.diff(stego_u8.astype(np.float64), axis=1)
    res_mae, res_max = float(np.mean(np.abs(r1 - r2))), float(np.max(np.abs(r1 - r2)))

    print("\n" + "=" * 70)
    print("EMPIRICAL STEGANALYSIS & UNDETECTABILITY EVALUATION")
    print("=" * 70)
    for name, rs in (("Cover Image", cover_rs), ("Stego Image", stego_rs)):
        print(f"    [{name}] R_M {rs['r_m']:.4f}% S_M {rs['s_m']:.4f}% R_-M {rs['r_neg_m']:.4f}% "
              f"S_-M {rs['s_neg_m']:.4f}% | est. payload p {rs['estimated_payload_p']:.4f}%")
    print(f"    RS Net Differential Payload Shift (dp): {rs_diff_p:+.4f}%")
    print(f"    Adjacent Pixel Residual MAE: {res_mae:.6f} | max {res_max:.4f}")
    spam = _feature_diff(calculate_spam_features, "SPAM", cover_u8, stego_u8)
    srm = _feature_diff(calculate_srm_features, "SRM", cover_u8, stego_u8)
    print("        Note: RS thresholds alone cannot establish undetectability.")
    print("=" * 70)
    return {
        "cover_rs": cover_rs, "stego_rs": stego_rs, "diff_p": rs_diff_p,
        "residual_mae": res_mae, "residual_max": res_max,
        "spam": {k: spam[k] for k in ("difference_l2", "difference_mae", "difference_max")},
        "srm": srm,
    }


def calculate_metrics(original, stego, embedded_bits, ber, run_steganalysis=True):
    original = np.asarray(original, dtype=np.float64)
    stego = np.asarray(stego, dtype=np.float64)
    diff = stego - original
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    psnr = float("inf") if mse <= 1e-16 else float(10.0 * np.log10(255.0 ** 2 / mse))
    ssim = float(structural_similarity(original, stego, data_range=255.0))
    fsim = float(calculate_fsim(original, stego))
    po, ps = _prob_hist(original), _prob_hist(stego)
    hist_sim = float(np.sum(np.sqrt(po * ps)))
    kl = float(np.sum(po * np.log(po / ps)))
    ent_o, ent_s = _entropy(original), _entropy(stego)
    npcr = float(np.mean(np.abs(diff) > 1e-12) * 100.0)
    uaci = float(np.mean(np.abs(diff) / 255.0) * 100.0)
    bpp = float(embedded_bits / int(original.size))
    max_change = float(np.max(np.abs(diff)))

    print("\n" + "=" * 70)
    print("IMAGE QUALITY METRICS  (8-BIT STEGO vs 8-BIT COVER)")
    print("=" * 70)
    for k, v in (("SSIM", f"{ssim:.6f}"), ("FSIM", f"{fsim:.6f}"), ("MSE", f"{mse:.6f}"),
                 ("MAE", f"{mae:.6f}"), ("RMSE", f"{rmse:.6f}"), ("PSNR", f"{psnr:.4f} dB"),
                 ("Original Entropy", f"{ent_o:.6f}"), ("Stego Entropy", f"{ent_s:.6f}"),
                 ("KL Divergence", f"{kl:.8f}"), ("Histogram Similarity", f"{hist_sim:.6f}"),
                 ("NPCR", f"{npcr:.4f} %"), ("UACI", f"{uaci:.6f} %"), ("BER", f"{ber:.6f}"),
                 ("BPP", f"{bpp:.6f}"), ("Maximum Absolute Change", f"{max_change:.1f}")):
        _row(k, v)
    print("=" * 70)
    steg = calculate_and_print_steganalysis(original, stego) if run_steganalysis else None
    return {
        "ssim": ssim, "fsim": fsim, "mse": mse, "rmse": rmse, "psnr": psnr, "mae": mae,
        "original_entropy": ent_o, "stego_entropy": ent_s, "kl_divergence": kl,
        "histogram_similarity": hist_sim, "npcr": npcr, "uaci": uaci, "ber": float(ber),
        "bpp": bpp, "maximum_absolute_change": max_change,
        "embedded_bits": int(embedded_bits), "steganalysis": steg,
    }


def save_experimental_visualizations(original_image, stego_image, output_dir):
    vis_dir = Path(output_dir) / "experimental_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    o = np.rint(np.clip(np.asarray(original_image, dtype=np.float64), 0, 255)).astype(np.uint8)
    s = np.rint(np.clip(np.asarray(stego_image, dtype=np.float64), 0, 255)).astype(np.uint8)
    diff = np.clip(np.abs(s.astype(np.float64) - o.astype(np.float64)) * 50.0, 0, 255).astype(np.uint8)
    plt.figure(figsize=(15, 5))
    for k, (img, title) in enumerate(((o, "Cover Image"), (s, "Stego Image"), (diff, "Amplified Difference")), 1):
        plt.subplot(1, 3, k)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(vis_dir / "cover_stego_difference.png", dpi=300, bbox_inches="tight")
    plt.close()
    ch = np.histogram(o, bins=256, range=(0, 256), density=True)[0]
    sh = np.histogram(s, bins=256, range=(0, 256), density=True)[0]
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(256), ch, label="Cover Image")
    plt.plot(np.arange(256), sh, label="Stego Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Normalized Frequency")
    plt.title("Cover and Stego Image Histogram Comparison")
    plt.xlim(0, 255)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / "cover_stego_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualizations saved in {vis_dir}")


def identify_payload_type(recovered_bytes):
    if len(recovered_bytes) >= 2 and recovered_bytes[:2] == b"\xff\xd8":
        return "image"
    try:
        recovered_bytes.decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        return "binary"


def calculate_secret_image_metrics(original_image, reconstructed_image, ber=0.0):
    original = np.asarray(original_image.convert("RGB"), dtype=np.uint8)
    reconstructed = np.asarray(reconstructed_image.convert("RGB"), dtype=np.uint8)
    if original.shape != reconstructed.shape:
        reconstructed_image = reconstructed_image.resize(
            (original_image.width, original_image.height), Image.Resampling.LANCZOS)
        reconstructed = np.asarray(reconstructed_image.convert("RGB"), dtype=np.uint8)
    of, rf = original.astype(np.float64), reconstructed.astype(np.float64)
    diff = rf - of
    mse = float(np.mean(diff ** 2))
    psnr = float("inf") if mse <= 0.0 else float(10.0 * np.log10(255.0 ** 2 / mse))
    ssim = float(structural_similarity(original, reconstructed, data_range=255, channel_axis=-1))
    po, pr = _prob_hist(original, 256), _prob_hist(reconstructed, 256)
    metrics = {
        "ssim": ssim, "mse": mse, "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(diff))), "psnr": psnr,
        "kl_divergence": float(np.sum(po * np.log(po / pr))),
        "histogram_similarity": float(np.sum(np.sqrt(po * pr))),
        "npcr": float(np.mean(original != reconstructed) * 100.0),
        "ber": float(ber), "maximum_absolute_change": float(np.max(np.abs(diff))),
    }
    print("\n" + "=" * 70)
    print("SECRET IMAGE QUALITY METRICS")
    print("=" * 70)
    _row("SSIM", f"{ssim:.6f}")
    _row("PSNR", f"{psnr:.4f} dB")
    _row("BER", f"{ber:.6f}")
    print("=" * 70)
    return metrics


def calculate_secret_image_psnr(sender_image, receiver_image):
    original = np.asarray(sender_image.convert("RGB"), dtype=np.float64)
    rec_img = receiver_image.convert("RGB")
    if original.shape != (rec_img.height, rec_img.width, 3):
        rec_img = rec_img.resize(sender_image.size, Image.Resampling.LANCZOS)
    mse = np.mean((original - np.asarray(rec_img, dtype=np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10.0 * np.log10((255.0 ** 2) / mse))


def plot_receiver_summary_figure(cover_image, stego_image, payload_type, metrics,
                                 secret_image=None, sender_secret_image=None,
                                 recovered_text=None, output_path=None,
                                 image_name="Cover Image", extra_save_path=None):
    cover_u8 = np.rint(np.clip(cover_image, 0.0, 255.0)).astype(np.uint8)
    stego_u8 = np.rint(np.clip(stego_image, 0.0, 255.0)).astype(np.uint8)
    fig = plt.figure(figsize=(16, 7.2))
    fig.suptitle(f"Image Name : {image_name.upper()}", fontsize=13, fontweight="bold", y=0.965)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(cover_u8, cmap="gray")
    ax.set_title("Restored Cover (bit-exact)", fontsize=11)
    ax.axis("off")
    total_bits = metrics.get("embedded_bits", "N/A")

    def secret_panel(position, img, title):
        a = fig.add_subplot(gs[0, position])
        if payload_type == "image" and img is not None:
            a.imshow(np.asarray(img.convert("RGB"), dtype=np.uint8))
            a.axis("off")
        else:
            a.set_facecolor("#ffffff")
            clean = (recovered_text or "").strip().replace("\r\n", " ").replace("\n", " ")
            footer = f"\n\n[Total Embedded Payload: {total_bits} bits]"
            snippet = (clean[:970].rstrip() + " ..." if len(clean) > 970 else clean) + footer
            a.text(0.038, 0.965, "\n".join(textwrap.wrap(snippet, width=47, replace_whitespace=False)),
                   ha="left", va="top", fontsize=7.5, color="#111111", family="monospace", linespacing=1.20)
            a.set_xlim(0, 1)
            a.set_ylim(0, 1)
            a.set_xticks([])
            a.set_yticks([])
        a.set_title(title, fontsize=11)

    secret_panel(1, sender_secret_image, "Secret (Sender)")
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(stego_u8, cmap="gray")
    ax.set_title("Stego Image", fontsize=11)
    ax.axis("off")
    secret_panel(3, secret_image, "Extracted Secret (Receiver)")

    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")
    try:
        extraction_score = max(0.0, min(100.0, (1.0 - float(metrics.get("ber"))) * 100.0))
    except (TypeError, ValueError):
        extraction_score = None

    def g(k):
        val = metrics.get(k, "N/A")
        if isinstance(val, (int, float, np.floating)):
            if k == "kl_divergence": return f"{val:.8f}"
            if k in ("psnr", "npcr", "secret_image_psnr"): return f"{val:.4f}"
            if k in ("mse", "mae", "rmse", "uaci", "ssim", "fsim", "bpp", "ber", "original_entropy",
                     "stego_entropy", "histogram_similarity"): return f"{val:.6f}"
            if k == "maximum_absolute_change": return f"{val:.1f}"
        return str(val)

    if payload_type == "image":
        ex_label, ex_value = "Extraction Score", (f"{extraction_score:.4f}%" if extraction_score is not None else "N/A")
        sp = metrics.get("secret_image_psnr", "N/A")
        sm_label, sm_value = "Secret Image PSNR", (f"{float(sp):.4f} dB" if sp != "N/A" else "N/A")
    else:
        ex_label = "Extraction Result"
        ex_value = "ALL DATA RECOVERED EXACTLY" if (extraction_score or 0) >= 100.0 else f"{extraction_score}"
        sm_label, sm_value = "Secret Data", payload_type.upper() + " Payload"

    steg = metrics.get("steganalysis") or {}
    rc = (steg.get("cover_rs") or {}).get("estimated_payload_p")
    rs = (steg.get("stego_rs") or {}).get("estimated_payload_p")
    rs_levels = f"{rc:.4f}% / {rs:.4f}%" if rc is not None and rs is not None else "N/A"
    dp = steg.get("diff_p")
    spam_l2 = (steg.get("spam") or {}).get("difference_l2")
    srm_l2 = (steg.get("srm") or {}).get("difference_l2")
    table_data = [
        ["PSNR (dB)", g("psnr"), "SSIM", g("ssim")],
        ["FSIM", g("fsim"), "MSE", g("mse")],
        ["MAE", g("mae"), "RMSE", g("rmse")],
        ["Original Entropy", g("original_entropy"), "Stego Entropy", g("stego_entropy")],
        ["KL Divergence", g("kl_divergence"), "Histogram Similarity", g("histogram_similarity")],
        ["NPCR (%)", g("npcr"), "UACI (%)", g("uaci")],
        ["BER", g("ber"), "RS Cover / Stego (p)", rs_levels],
        ["RS Shift (dp)", f"{dp:+.4f}%" if dp is not None else "N/A", "SPAM (L2)",
         f"{spam_l2:.6f}" if spam_l2 is not None else "N/A"],
        ["SRM (L2)", f"{srm_l2:.6f}" if srm_l2 is not None else "N/A", "Max Absolute Change", g("maximum_absolute_change")],
        [ex_label, ex_value, sm_label, sm_value],
        ["Payload Type", payload_type.upper(), "Total Embedded Bits", g("embedded_bits")],
    ]
    table = ax_table.table(cellText=table_data, colLabels=["Metric", "Value", "Metric", "Value"],
                           loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 1.25)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for p in (output_path, extra_save_path):
        if p:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=300, bbox_inches="tight")
            print(f"[SAVED] {p}")
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def print_pixel_preservation_stats(cover_image, stego_image):
    cover = np.asarray(cover_image, dtype=np.int32)
    stego = np.asarray(stego_image, dtype=np.int32)
    total = int(cover.size)
    changed = int(np.count_nonzero(cover != stego))
    print("\n" + "=" * 70)
    print("CARRIER PIXEL PRESERVATION ANALYSIS")
    print("=" * 70)
    print(f"{'Total Image Pixels':<30}: {total:,}")
    print(f"{'Modified Pixels':<30}: {changed:,} ({changed / total * 100:.4f}%)")
    print(f"{'Untouched Cover Pixels':<30}: {total - changed:,} ({(total - changed) / total * 100:.4f}%)")
    print("=" * 70)


def sender():
    SENDER_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    message_metadata = load_message_metadata()
    qkd_blueprint = load_qkd_blueprint()
    plan = load_authoritative_plan(required=False)
    qrng = QRNG(QRNG_SEED, "alice")

    chunk_ids = sorted(chunks.keys())
    chunk_lens = [chunks[cid]["bit_length"] for cid in chunk_ids]
    N = total_stream_bits(chunk_lens)
    message_bits_count = sum(chunk_lens)
    secret_bytes = bytes_from_bits("".join(chunks[c]["bits"] for c in chunk_ids))

    blueprint_by_chunk = {
        item["chunk_id"]: item
        for item in qkd_blueprint["hierarchy"].get("subkeys", [])
        if "chunk_id" in item
    }

    print("=" * 70)
    print("STEGAQENTROPY ML+PE REVERSIBLE PIXEL EMBEDDING - SENDER")
    print("=" * 70)
    print(f"Chunks loaded       : {len(chunks)}")
    print(f"Message bit length  : {message_bits_count}")
    print(f"Total stream bits   : {N} (meta {META_BITS} + QKD {len(chunk_ids) * SUBKEY_BITS if EMBED_SUBKEYS else 0} + body {message_bits_count})")
    print(f"ML regions loaded   : {len(plan)}")
    print(f"Spread tile         : {SPREAD_TILE if SPREAD_TILE else 'global (fewest changes)'}")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"Waiting for receiver: {HOST}:{PORT}")
    connection, address = server.accept()
    connection.settimeout(600.0)
    print(f"Receiver connected: {address}")

    with connection:
        send_packet(connection, {
            "type": "hello",
            "chunk_ids": chunk_ids,
            "hierarchy_id": qkd_blueprint["hierarchy"].get("hierarchy_id", "QKH")
        })
        if receive_packet(connection)["type"] != "hello_ack":
            raise RuntimeError("Receiver handshake failed.")

        master_result = qkd_alice(connection, qrng, "MASTER", MASTER_KEY_BITS, MASTER_QUBITS)
        master_key = master_result["key"]

        subkeys = {}
        for cid in chunk_ids:
            if cid not in blueprint_by_chunk:
                raise RuntimeError(f"QKD blueprint has no subkey configuration for {cid}.")
            subkeys[cid] = qkd_alice(connection, qrng, f"SUBKEY_{cid}", SUBKEY_BITS, SUBKEY_QUBITS)["key"]

        stream = build_stream(chunks, chunk_lens, chunk_ids, subkeys, master_key)
        cover_u8 = build_cover_u8()
        bitmap = plan_to_bitmap(plan, cover_u8.shape) if plan else "0" * (side_nbits(cover_u8.shape) - 16)
        ml = mask_from_bitmap(bitmap, cover_u8.shape) if "1" in bitmap else None
        tgt = target_psnr(N, cover_u8.shape)

        t = time.time()
        stego_u8, info = embed_with_side_info(cover_u8, stream, master_key, bitmap, ml)
        aux_bits = side_nbits(cover_u8.shape)
        diff = stego_u8.astype(np.int64) - cover_u8.astype(np.int64)
        max_change = int(np.max(np.abs(diff)))
        info["changes"] = int(np.count_nonzero(diff))
        info["bits_per_change"] = (N + aux_bits) / max(info["changes"], 1)
        if max_change > 1:
            raise RuntimeError(f"Max absolute pixel change is {max_change} (must be 1).")
        Image.fromarray(stego_u8).save(STEGO_IMAGE, format="PNG")
        saved_u8 = np.asarray(Image.open(STEGO_IMAGE), dtype=np.uint8)

        # Receiver order: undo side-info stage first (no mask needed), then payload
        bm_rx, img_b, R_rx = extract_side_info(saved_u8, master_key)
        if bm_rx != bitmap:
            raise RuntimeError("ML side-info self-check failed.")
        ml_rx = mask_from_bitmap(bm_rx, cover_u8.shape) if "1" in bm_rx else None
        bits, restored, cfg = pe_extract(
            img_b, N, master_key, ml_rx,
            make_verifier(chunk_lens, chunk_ids, subkeys, master_key), R_rx
        )
        if bits != stream or not np.array_equal(restored, cover_u8):
            raise RuntimeError("Reversibility self-check failed.")

        print("\n" + "=" * 70)
        print("ML + PE REVERSIBLE PIXEL EMBEDDING")
        print("=" * 70)
        print(f"Payload bits            : {N}")
        print(f"ML side-info bits       : {aux_bits} ({bitmap.count('1')} ML cells, grid {ML_GRID})")
        print(f"Total embedded bits     : {N + aux_bits}")
        print(f"Max abs pixel change    : {max_change}")
        print(f"Changed-pixel limit     : {MAX_CHANGED_PIXELS} -> "
              f"{'OK' if info['changes'] <= MAX_CHANGED_PIXELS else 'EXCEEDED'}")
        print(f"Config (not sent)       : {info['config']}")
        print(f"Pixels changed          : {info['changes']}")
        print(f"Bits per changed pixel  : {info['bits_per_change']:.3f}")
        print(f"Search time             : {time.time() - t:.1f}s")
        print("[SELF-CHECK] payload exact + cover restored bit-exactly from stego alone")
        print("=" * 70)

        save_selected_regions_plot(plan)
        original_image = cover_u8.astype(np.float64)
        stego_image = saved_u8.astype(np.float64)
        m8 = metrics_8bit(cover_u8, saved_u8)
        print(f"PSNR {m8['psnr']:.4f} dB (target {tgt:.2f}) | SSIM {m8['ssim']:.6f} | FSIM {m8['fsim']:.6f}")
        save_experimental_visualizations(original_image, stego_image, OUTPUT_DIR)
        Image.fromarray(stego_u8).save(STEGO_VIEW_IMAGE)
        total_embedded_bits = N + aux_bits
        metrics = calculate_metrics(
            original_image, stego_image, total_embedded_bits, 0.0,
            run_steganalysis=False
        )

        time.sleep(1)
        send_packet(connection, {
            "type": "stego_transmission",
            "filename": STEGO_IMAGE.name,
            "chunk_ids": chunk_ids,
            "chunk_lens": chunk_lens,
            "payload_type": identify_payload_type(secret_bytes),
            "data": STEGO_IMAGE.read_bytes().hex(),
            "sha256": hashlib.sha256(STEGO_IMAGE.read_bytes()).hexdigest(),
        })

        receiver_result = receive_packet(connection)
        if receiver_result["type"] != "receiver_result":
            raise RuntimeError("Receiver acknowledgement not received.")
        if receiver_result.get("sha256") != hashlib.sha256(secret_bytes).hexdigest():
            raise RuntimeError("Receiver hash mismatch: message not recovered exactly.")
        ber = 0.0   # exact hash match => BER 0 (measured at sender)
        metrics["ber"] = ber

        save_json(PLAN_JSON, {
            "message_metadata": message_metadata,
            "payload_bits": N,
            "ml_side_info_bits": aux_bits,
            "max_abs_change": max_change,
            "chunk_lens": chunk_lens,
            "pe_config": list(info["config"]),
            "changes": info["changes"],
            "qkd_master_qber": master_result["qber"],
            "metrics": metrics,
            "receiver_status": receiver_result.get("status"),
        })
        save_json(METRICS_JSON, metrics)
        print_pixel_preservation_stats(original_image, stego_image)
        pm = evaluate_structural_pattern_leakage(cover_u8, stego_u8)
        print(f"EAR {pm['edge_alignment_ratio_pct']:.2f}% | SCC {pm['structural_contour_correlation']:.6f} | "
              f"max chain {pm['max_chain_component_size']} | Moran's I {pm['morans_i']:.6f}")
    server.close()


def receiver():
    RECEIVER_DIR.mkdir(parents=True, exist_ok=True)
    qrng = QRNG(QRNG_SEED, "bob")

    print("=" * 70)
    print("STEGAQENTROPY ML+PE REVERSIBLE PIXEL EMBEDDING - RECEIVER")
    print("=" * 70)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for _ in range(60):
        try:
            sock.connect((HOST, PORT))
            break
        except ConnectionRefusedError:
            time.sleep(1)
    sock.settimeout(600.0)

    with sock:
        hello = receive_packet(sock)
        if hello["type"] != "hello":
            raise RuntimeError("Invalid sender handshake.")
        chunk_ids = hello["chunk_ids"]
        send_packet(sock, {"type": "hello_ack"})

        master_result = qkd_bob(sock, qrng, MASTER_KEY_BITS)
        master_key = master_result["key"]

        subkeys = {}
        for _ in range(len(chunk_ids)):
            r = qkd_bob(sock, qrng, SUBKEY_BITS)
            if not r["session_id"].startswith("SUBKEY_"):
                raise RuntimeError("Unexpected QKD session ID format.")
            cid = r["session_id"].split("_", 1)[1]
            if cid not in chunk_ids:
                raise RuntimeError(f"Unexpected subkey session for {cid}.")
            subkeys[cid] = r["key"]

        pkt = receive_packet(sock)
        if pkt["type"] != "stego_transmission":
            raise RuntimeError("Stego image transmission not received.")

        stego_bytes = bytes.fromhex(pkt["data"])
        if not secrets.compare_digest(hashlib.sha256(stego_bytes).hexdigest(), pkt["sha256"]):
            raise RuntimeError("Stego image transmission integrity check failed.")

        chunk_lens = pkt["chunk_lens"]
        N = total_stream_bits(chunk_lens)
        safe_name = Path(pkt.get("filename", "received_stego.png")).name
        stego_path = RECEIVER_DIR / safe_name
        stego_path.write_bytes(stego_bytes)
        stego_u8 = np.asarray(Image.open(io.BytesIO(stego_bytes)), dtype=np.uint8)

        # Undo the ML-map stage first (needs no mask), then decode the payload
        bm_rx, stego_b, R_rx = extract_side_info(stego_u8, master_key)
        print(f"ML cells from stego : {bm_rx.count('1')}")
        ml = mask_from_bitmap(bm_rx, stego_u8.shape) if "1" in bm_rx else None
        t = time.time()
        bits, restored_u8, cfg = pe_extract(
            stego_b, N, master_key, ml,
            make_verifier(chunk_lens, chunk_ids, subkeys, master_key), R_rx
        )
        message_bits, crc_ok = parse_stream(bits, chunk_lens, chunk_ids, subkeys, master_key)
        print(f"Config found by trial decoding: {cfg}  ({time.time() - t:.1f}s)")
        if not crc_ok:
            raise RuntimeError("Message CRC32 mismatch.")

        recovered_bytes = bytes_from_bits(message_bits)
        EXTRACTED_MESSAGE.write_bytes(recovered_bytes)
        Image.fromarray(restored_u8).save(RECEIVER_DIR / "restored_cover_image.png")

        ber = 0.0  # CRC32 verified here; sender confirms via SHA-256 and records BER
        payload_type = pkt.get("payload_type") or identify_payload_type(recovered_bytes)
        original_image = restored_u8.astype(np.float64)
        stego_image = stego_u8.astype(np.float64)
        aux_bits = side_nbits(stego_u8.shape)
        total_embedded_bits = N + aux_bits

        metrics = calculate_metrics(
            original_image, stego_image, total_embedded_bits, ber,
            run_steganalysis=True
        )

        rec_img = None
        if payload_type == "image":
            rec_img = Image.open(io.BytesIO(recovered_bytes)).convert("RGB")
            rec_img.load()
            rec_img.save(RECEIVER_DIR / "received_extracted_image.png")

        send_packet(sock, {
            "type": "receiver_result",
            "status": "SUCCESS",
            "ber": ber,
            "recovered_bits": len(message_bits),
            "sha256": hashlib.sha256(recovered_bytes).hexdigest(),
            "metrics": metrics,
        })
        save_json(RECEIVER_METRICS_JSON, {"cover_metrics": metrics})

        print(f"\nQKD master QBER     : {master_result['qber']:.6f}")
        print(f"Payload bits        : {N} | CRC32 ok | BER {ber:.6f}")
        print("Cover restored bit-exactly from stego alone: PSNR(restored cover) = inf")
        print(f"Stego PSNR          : {metrics['psnr']:.4f} dB (target {target_psnr(N, stego_u8.shape):.2f})")

        recovered_str = None
        if payload_type == "text":
            print("\n" + "=" * 70)
            print("RECOVERED MESSAGE:")
            print("=" * 70)
            print(recovered_bytes.decode("utf-8"))
            print("=" * 70)
            recovered_str = recovered_bytes.decode("utf-8", errors="replace")

        tag = "cover"
        profile_path = BASE_DIR / "output" / "image_acquisition" / "image_profile.json"
        if profile_path.exists():
            try:
                prof = load_json(profile_path)
                raw_path = (prof.get("image_path") or prof.get("cover_image_path") or
                            prof.get("original_image_path") or prof.get("filename"))
                if raw_path:
                    tag = Path(raw_path).stem.replace("_", " ").title()
            except Exception:
                pass

        plot_receiver_summary_figure(
            cover_image=original_image,
            stego_image=stego_image,
            payload_type=payload_type,
            metrics=metrics,
            secret_image=rec_img,
            sender_secret_image=None,
            recovered_text=recovered_str,
            output_path=RECEIVER_DIR / "receiver_summary_plot.png",
            image_name=tag,
            extra_save_path=BASE_DIR / "experiment_results" / f"{tag.lower().replace(' ', '_')}_result_summary.png",
        )


def launch_sender_receiver():
    script, py = Path(__file__).resolve(), sys.executable
    subprocess.Popen([
        "wt.exe", "new-tab", "--title", "Spatial Opt Sender",
        "cmd", "/k", f'"{py}" "{script}" sender'
    ], cwd=BASE_DIR)
    time.sleep(2)
    subprocess.Popen([
        "wt.exe", "-w", "0", "split-pane", "-V", "--title", "Spatial Opt Receiver",
        "cmd", "/k", f'"{py}" "{script}" receiver'
    ], cwd=BASE_DIR)


def main():
    if len(sys.argv) == 1:
        launch_sender_receiver()
    elif sys.argv[1].lower() == "sender":
        sender()
    elif sys.argv[1].lower() == "receiver":
        receiver()
    else:
        raise SystemExit("Usage: python adap_dwt_pee.py [sender|receiver]")


if __name__ == "__main__":
    main()

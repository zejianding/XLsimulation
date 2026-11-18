"""Batched XL theoretical grids and single-scan targets for candidate rescoring."""

from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np
import torch

from build_xl_spectrum_dataset import (
    CLIP_INTENSITY,
    LINKER_MASSES,
    MIN_VALID_MZ,
    NUM_CHANNELS,
    PEAK_TOLERANCE,
    RESIDUE_MODS,
    TOKEN_OFFSETS,
    normalize_target_intensity,
)
from sequence_token_map import _MZ_EQ_TOL
from token_table import AA_MASS, H2O, MAX_PEPTIDE_LEN, NUM_TOKENS, PROTON, RESIDUE_EVENT_TOKEN_IDS


_LOOKUP_BINS_PER_DA = 16


def _kernel_source():
    residue_masses = np.zeros(256, dtype=np.float64)
    for residue, mass in AA_MASS.items():
        residue_masses[ord(residue)] = mass
    residue_events = np.zeros(NUM_TOKENS, dtype=np.int8)
    residue_events[list(RESIDUE_EVENT_TOKEN_IDS)] = 1

    def declaration(name, dtype, values):
        items = ", ".join(str(value) for value in values)
        return f"__device__ __constant__ {dtype} {name}[] = {{{items}}};\n"

    # At fixed charge, a positive residue offset can reach at most three
    # adjacent cleavage intervals. Both accepted modifications add mass.
    neighbor_count = int((max(AA_MASS.values()) + _MZ_EQ_TOL * 3) / min(AA_MASS.values()))
    constants = (
        f"#define CHANNELS {NUM_CHANNELS}\n"
        f"#define TOKENS {NUM_TOKENS}\n"
        f"#define MAX_LENGTH {MAX_PEPTIDE_LEN}\n"
        f"#define MAX_NEIGHBORS {neighbor_count}\n"
        f"#define N_BINS_PER_DA {_LOOKUP_BINS_PER_DA}\n"
        f"#define LINKER_MASS {LINKER_MASSES['BS3']}\n"
        f"#define CAM {RESIDUE_MODS[2]}\n"
        f"#define OX {RESIDUE_MODS[21]}\n"
        f"#define PROTON_MASS {PROTON}\n"
        f"#define WATER_MASS {H2O}\n"
        f"#define MZ_TOL {PEAK_TOLERANCE}\n"
        f"#define FULL_TOL {_MZ_EQ_TOL}\n"
        f"#define MIN_MZ {MIN_VALID_MZ}\n"
        + declaration("residue_masses", "double", residue_masses)
        + declaration("token_offsets", "double", TOKEN_OFFSETS[:, :NUM_CHANNELS].flatten())
        + declaration("residue_events", "char", residue_events)
    )
    return constants + r"""
extern "C" __global__ void build_base(
    const unsigned char* aa, const bool* ox, const long long* lengths,
    const long long* sites, int sequence_width, int charge, double* base
) {
    const int pair = blockIdx.x;
    const int cleavages = sequence_width - 1;
    const int first_chain = pair * 2;
    __shared__ double prefixes[2][MAX_LENGTH];
    __shared__ double totals[2];

    // Sequential sums and separate operations reproduce the Python builder.
    if (threadIdx.x < 2) {
        const int p = threadIdx.x;
        const int chain = first_chain + p;
        const int start = chain * sequence_width;
        double total = 0.0;
        for (int i = 0; i < lengths[chain]; i++) {
            const unsigned char residue = aa[start + i];
            double mass = residue_masses[residue];
            if (residue == 'C') mass += CAM;
            if (ox[start + i]) mass += OX;
            total += mass;
            prefixes[p][i] = total;
        }
        totals[p] = total;
    }
    __syncthreads();

    for (int index = threadIdx.x; index < 2 * cleavages * CHANNELS; index += blockDim.x) {
        const int channel = index % CHANNELS;
        const int p = index / (cleavages * CHANNELS);
        const int row = (index / CHANNELS) % cleavages;
        const int chain = first_chain + p;
        const int z = channel % (CHANNELS / 2) + 1;
        double value = 0.0;
        if (row < lengths[chain] - 1 && z <= charge) {
            const bool b = channel < CHANNELS / 2;
            const bool contains_link = b == (row + 1 >= sites[chain]);
            const double prefix = prefixes[p][row];
            const double mass = b ? prefix : (totals[p] - prefix) + WATER_MASS;
            value = (mass + z * PROTON_MASS) / z;
            if (contains_link) {
                const double partner_mass = totals[1 - p] + WATER_MASS;
                value += (partner_mass + LINKER_MASS) / z;
            }
        }
        base[pair * 2 * cleavages * CHANNELS + index] = value;
    }
}

extern "C" __global__ void project_grid(
    const double* base, const long long* lengths, int cleavages, int charge,
    const double* peak_mz, const float* peak_intensity, const int* bins,
    int first_bin, int last_bin, int peak_count, double* grid, bool* valid, float* target
) {
    const int chain = blockIdx.y;
    const int chain_cells = cleavages * TOKENS * CHANNELS;
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= chain_cells) return;
    const int channel = cell % CHANNELS;
    const int offset_index = cell % (TOKENS * CHANNELS);
    const int token = offset_index / CHANNELS;
    const int row = cell / (TOKENS * CHANNELS);
    const int out = chain * chain_cells + cell;
    const int length = lengths[chain];
    const int z = channel % (CHANNELS / 2) + 1;

    if (row >= length - 1 || z > charge) {
        grid[out] = 0.0;
        valid[out] = false;
        target[out] = 0.0f;
        return;
    }
    const double* current = base + (chain * cleavages + row) * CHANNELS + channel;
    const double offset = token_offsets[offset_index];
    const double mz = *current + offset;
    grid[out] = mz;
    valid[out] = isfinite(mz) && mz >= MIN_MZ;
    target[out] = 0.0f;
    if (!valid[out] || !peak_count) return;

    if (residue_events[token]) {
        const int direction = ((offset > 0.0) == (channel < CHANNELS / 2)) ? 1 : -1;
        bool supported = false;
        #pragma unroll
        for (int distance = 1; distance <= MAX_NEIGHBORS; distance++) {
            const int next = row + direction * distance;
            if (next >= 0 && next < length - 1) {
                const double delta = mz - current[direction * distance * CHANNELS];
                supported |= fabs(delta) <= FULL_TOL;
            }
        }
        if (!supported) return;
    }

    const double lower = mz - MZ_TOL;
    const double upper = mz + MZ_TOL;
    const int bucket = __double2int_rd(lower * N_BINS_PER_DA);
    if (bucket > last_bin) return;
    const int first = bucket <= first_bin ? 0 : bins[bucket - first_bin];
    float strongest = 0.0f;
    for (int peak = first; peak < peak_count && peak_mz[peak] <= upper; peak++) {
        if (peak_mz[peak] >= lower) strongest = fmaxf(strongest, peak_intensity[peak]);
    }
    target[out] = strongest;
}
"""


class _PeakLookup(NamedTuple):
    mz: torch.Tensor
    intensity: torch.Tensor
    bins: torch.Tensor
    first_bin: int
    last_bin: int


class XLProjection:
    """Build the model grid and a full-constrained target on the CUDA device."""

    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Batched XL grid projection requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        # CuPy is used solely to launch CUDA kernels on the current Torch stream.
        os.environ.setdefault("CUPY_CACHE_IN_MEMORY", "1")
        import cupy

        self._cupy = cupy
        self._module = cupy.RawModule(code=_kernel_source(), options=("--fmad=false",))

    def prepare_spectrum(self, mz, intensity):
        """Index a single experimental centroided spectrum for interval queries."""
        mz = np.asarray(mz, dtype=np.float64)
        intensity = np.asarray(intensity, dtype=np.float64)
        keep = np.isfinite(mz) & (mz > 0.0) & np.isfinite(intensity) & (intensity > 0.0)
        mz, intensity = mz[keep], intensity[keep]
        order = np.argsort(mz, kind="stable")
        mz, intensity = mz[order], intensity[order]
        # Clipping and the square root are nondecreasing, so taking the strongest
        # already-normalized peak is identical to normalizing after the max.
        intensity, _ = normalize_target_intensity(intensity, method="sqrt_clip", clip_value=CLIP_INTENSITY)
        first = int(np.floor(mz[0] * _LOOKUP_BINS_PER_DA)) if mz.size else 0
        last = int(np.floor(mz[-1] * _LOOKUP_BINS_PER_DA)) if mz.size else 0
        bin_edges = np.arange(first, last + 1, dtype=np.float64) / _LOOKUP_BINS_PER_DA
        starts = np.searchsorted(mz, bin_edges).astype(np.int32)
        return _PeakLookup(
            torch.from_numpy(mz).to(self.device),
            torch.from_numpy(intensity).to(self.device),
            torch.from_numpy(starts).to(self.device),
            first,
            last,
        )

    def project(self, aa, ox, lengths, sites, charge, peak_state):
        """Project payloads with ASCII residues, fixed CAM, and bool M oxidation."""
        aa, ox, lengths, sites = (tensor.contiguous() for tensor in (aa, ox, lengths, sites))
        batch_size, _, sequence_width = aa.shape
        cleavages = sequence_width - 1
        shape = (batch_size, 2, cleavages, NUM_TOKENS, NUM_CHANNELS)
        base = torch.empty((batch_size, 2, cleavages, NUM_CHANNELS), device=self.device, dtype=torch.float64)
        grid = torch.empty(shape, device=self.device, dtype=torch.float64)
        valid = torch.empty(shape, device=self.device, dtype=torch.bool)
        target = torch.empty(shape, device=self.device, dtype=torch.float32)
        if not batch_size:
            return grid, valid, target

        cupy = self._cupy
        threads = 128
        chain_cells = cleavages * NUM_TOKENS * NUM_CHANNELS
        stream = torch.cuda.current_stream(self.device).cuda_stream
        with cupy.cuda.Device(self.device.index), cupy.cuda.ExternalStream(stream):
            as_cupy = cupy.from_dlpack
            self._module.get_function("build_base")(
                (batch_size,), (threads,),
                (as_cupy(aa), as_cupy(ox), as_cupy(lengths), as_cupy(sites),
                 np.int32(sequence_width), np.int32(charge), as_cupy(base)),
            )
            self._module.get_function("project_grid")(
                ((chain_cells + threads - 1) // threads, 2 * batch_size), (threads,),
                (as_cupy(base), as_cupy(lengths), np.int32(cleavages), np.int32(charge),
                 as_cupy(peak_state.mz), as_cupy(peak_state.intensity), as_cupy(peak_state.bins),
                 np.int32(peak_state.first_bin), np.int32(peak_state.last_bin), np.int32(peak_state.mz.numel()),
                 as_cupy(grid), as_cupy(valid), as_cupy(target)),
            )
        return grid, valid, target

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Copied from NVRadar backprojection CUDA kernels
// (modules/edge/python/nvradar/edge/operators/backprojection_image_formation/bp_kernels.py)
// at SHA 37c1cb29355bcfe256ae728f51dd47071847d949. OSRB relicensing of the
// copies is recorded in this operator's OPEN_ITEMS.md. Do not copy
// backprojection_kernel (legacy float bin/w). Do not share a phase LUT
// with MatX — the fills differ.

#pragma once

namespace holoscan::radar::detail {

// Golden double-range / double-interp reference. Accumulator is still
// complex64 (NVRadar writes float* simg). vph is double*.
//
// img_width is the fast (u) axis; img_height is v. Do not inherit
// NVRadar's Python naming that set img_width = shape[0].
__global__ void backprojection_kernel_fp64_ref(
    int fcount, int img_width, int img_height, const int Np,
    const int phase_center_stride, const double rmin, const double dr,
    const double pc_partial, float* __restrict__ simg,
    const double* __restrict__ fvec, const double* __restrict__ ximg_grid,
    const double* __restrict__ yimg_grid, const double* __restrict__ zimg_grid,
    const double* __restrict__ phase_center, const double* __restrict__ R0,
    const double* __restrict__ vph, int Nfft) {
  (void)fcount;
  (void)fvec;
  const int xi = blockIdx.x * blockDim.x + threadIdx.x;
  const int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if ((xi >= img_width) || (yi >= img_height)) {
    return;
  }
  const int image_loc = yi * img_width + xi;
  const double x = ximg_grid[image_loc];
  const double y = yimg_grid[image_loc];
  const double z = zimg_grid[image_loc];
  float acc_r = 0.0f;
  float acc_i = 0.0f;
  for (int p = 0; p < Np; ++p) {
    const int pulse_offset = p * Nfft * 2;
    const double px = phase_center[p];
    const double py = phase_center[p + phase_center_stride];
    const double pz = phase_center[p + phase_center_stride * 2];
    const double pR0 = R0[p];
    double R = sqrt((x - px) * (x - px) + (y - py) * (y - py) +
                    (z - pz) * (z - pz));
    R = R - pR0;
    const double bin = (R - rmin) / dr;
    if ((bin < 0) || (bin >= Nfft - 1)) {
      continue;
    }
    const int b1 = static_cast<int>(floor(bin));
    const int b2 = b1 + 1;
    const double w = bin - floor(bin);
    const double v1r = vph[pulse_offset + b1 * 2];
    const double v1i = vph[pulse_offset + b1 * 2 + 1];
    const double v2r = vph[pulse_offset + b2 * 2];
    const double v2i = vph[pulse_offset + b2 * 2 + 1];
    const double vr = v1r + w * (v2r - v1r);
    const double vi = v1i + w * (v2i - v1i);
    const double mfr = cos(pc_partial * R);
    const double mfi = sin(pc_partial * R);
    acc_r += static_cast<float>(vr * mfr - vi * mfi);
    acc_i += static_cast<float>(vr * mfi + vi * mfr);
  }
  simg[2 * image_loc] += acc_r;
  simg[2 * image_loc + 1] += acc_i;
}

// Default operator backend: NVRadar Taylor refined-throughput
// (anchor_taylor_phi32_vphf_drinvd). float* vph is a view on complex64.
#define HOLOSCAN_RADAR_BP_BLOCK_SIDE 16
#define HOLOSCAN_RADAR_BP_BLOCK_PIXELS \
  (HOLOSCAN_RADAR_BP_BLOCK_SIDE * HOLOSCAN_RADAR_BP_BLOCK_SIDE)

__global__ void
backprojection_kernel_fp64_phiwrap_f32mf_f32vph_anchor_taylor_phi32_vphf_drinvd(  // NOLINT
    int fcount, int img_width, int img_height, const int Np,
    const int phase_center_stride, const double rmin,
    const double dr_inv_d, const float dr_inv, const double pc_partial,
    float* __restrict__ simg, const double* __restrict__ fvec,
    const double* __restrict__ ximg_grid,
    const double* __restrict__ yimg_grid,
    const double* __restrict__ zimg_grid,
    const double* __restrict__ phase_center, const double* __restrict__ R0,
    const float* __restrict__ vph, int Nfft) {
  (void)fcount;
  (void)fvec;
  const double TWOPI_D = 6.283185307179586476925286766559;
  const double INV_TWOPI_D = 0.15915494309189534560822210096757;
  const float TWOPI_F = 6.2831853071795865f;
  const float INV_TWOPI_F = 0.15915494309189534f;

  __shared__ int s_anchor_int[HOLOSCAN_RADAR_BP_BLOCK_PIXELS];
  __shared__ float s_anchor_frac[HOLOSCAN_RADAR_BP_BLOCK_PIXELS];
  __shared__ float s_ux[HOLOSCAN_RADAR_BP_BLOCK_PIXELS];
  __shared__ float s_uy[HOLOSCAN_RADAR_BP_BLOCK_PIXELS];
  __shared__ float s_uz[HOLOSCAN_RADAR_BP_BLOCK_PIXELS];
  __shared__ float s_inv_R[HOLOSCAN_RADAR_BP_BLOCK_PIXELS];
  __shared__ float s_phi_anchor[HOLOSCAN_RADAR_BP_BLOCK_PIXELS];
  __shared__ double s_ax_d;
  __shared__ double s_ay_d;
  __shared__ double s_az_d;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int xi = blockIdx.x * HOLOSCAN_RADAR_BP_BLOCK_SIDE + threadIdx.x;
  const int yi = blockIdx.y * HOLOSCAN_RADAR_BP_BLOCK_SIDE + threadIdx.y;
  const bool in_image = (xi < img_width) && (yi < img_height);
  const int image_loc = yi * img_width + xi;

  if (tid == 0) {
    const int ax = blockIdx.x * HOLOSCAN_RADAR_BP_BLOCK_SIDE +
                   HOLOSCAN_RADAR_BP_BLOCK_SIDE / 2;
    const int ay = blockIdx.y * HOLOSCAN_RADAR_BP_BLOCK_SIDE +
                   HOLOSCAN_RADAR_BP_BLOCK_SIDE / 2;
    const int ax_c = ax < img_width ? ax : (img_width - 1);
    const int ay_c = ay < img_height ? ay : (img_height - 1);
    const int anchor_idx = ay_c * img_width + ax_c;
    s_ax_d = ximg_grid[anchor_idx];
    s_ay_d = yimg_grid[anchor_idx];
    s_az_d = zimg_grid[anchor_idx];
  }
  __syncthreads();

  double x_d = 0.0;
  double y_d = 0.0;
  double z_d = 0.0;
  if (in_image) {
    x_d = ximg_grid[image_loc];
    y_d = yimg_grid[image_loc];
    z_d = zimg_grid[image_loc];
  }
  const float dx_a = in_image ? static_cast<float>(x_d - s_ax_d) : 0.0f;
  const float dy_a = in_image ? static_cast<float>(y_d - s_ay_d) : 0.0f;
  const float dz_a = in_image ? static_cast<float>(z_d - s_az_d) : 0.0f;
  const float pc_partial_f = static_cast<float>(pc_partial);

  float acc_r = 0.0f;
  float acc_i = 0.0f;

  for (int chunk = 0; chunk < Np; chunk += HOLOSCAN_RADAR_BP_BLOCK_PIXELS) {
    const int remain = Np - chunk;
    const int chunk_size = remain < HOLOSCAN_RADAR_BP_BLOCK_PIXELS
                               ? remain
                               : HOLOSCAN_RADAR_BP_BLOCK_PIXELS;
    if (tid < chunk_size) {
      const int p = chunk + tid;
      const double pcx = phase_center[p];
      const double pcy = phase_center[p + phase_center_stride];
      const double pcz = phase_center[p + phase_center_stride * 2];
      const double R0_p_d = R0[p];
      const double dxa = s_ax_d - pcx;
      const double dya = s_ay_d - pcy;
      const double dza = s_az_d - pcz;
      const double R_anchor_pure =
          sqrt(dxa * dxa + dya * dya + dza * dza);
      const double R_anchor = R_anchor_pure - R0_p_d;
      const double bin_anchor = (R_anchor - rmin) * dr_inv_d;
      const double bin_floor = floor(bin_anchor);
      const double inv_R_d = 1.0 / R_anchor_pure;
      const double phi_anchor_d = pc_partial * R_anchor;
      const double k_a = trunc(phi_anchor_d * INV_TWOPI_D);
      const double phi_anchor_w = fma(-k_a, TWOPI_D, phi_anchor_d);
      s_anchor_int[tid] = static_cast<int>(bin_floor);
      s_anchor_frac[tid] = static_cast<float>(bin_anchor - bin_floor);
      s_ux[tid] = static_cast<float>(dxa * inv_R_d);
      s_uy[tid] = static_cast<float>(dya * inv_R_d);
      s_uz[tid] = static_cast<float>(dza * inv_R_d);
      s_inv_R[tid] = static_cast<float>(inv_R_d);
      s_phi_anchor[tid] = static_cast<float>(phi_anchor_w);
    }
    __syncthreads();

    if (in_image) {
#pragma unroll 4
      for (int j = 0; j < chunk_size; ++j) {
        const float ux = s_ux[j];
        const float uy = s_uy[j];
        const float uz = s_uz[j];
        const float invR = s_inv_R[j];
        const float dot = ux * dx_a + uy * dy_a + uz * dz_a;
        const float dsq = dx_a * dx_a + dy_a * dy_a + dz_a * dz_a;
        const float perp_sq = dsq - dot * dot;
        const float dR =
            dot + perp_sq * 0.5f * invR * (1.0f - dot * invR);
        const float dbin = dR * dr_inv;
        const float bin_loc = s_anchor_frac[j] + dbin;
        const float bf_floor = floorf(bin_loc);
        const int b1 = s_anchor_int[j] + static_cast<int>(bf_floor);
        const float w = bin_loc - bf_floor;
        if ((b1 < 0) || (b1 >= Nfft - 1)) {
          continue;
        }
        const int p = chunk + j;
        const int pulse_offset = p * Nfft * 2;
        const float v1r = vph[pulse_offset + b1 * 2];
        const float v1i = vph[pulse_offset + b1 * 2 + 1];
        const float v2r = vph[pulse_offset + b1 * 2 + 2];
        const float v2i = vph[pulse_offset + b1 * 2 + 3];
        const float vr = v1r + w * (v2r - v1r);
        const float vi = v1i + w * (v2i - v1i);
        const float dphi = pc_partial_f * dR;
        const float phi_total = s_phi_anchor[j] + dphi;
        const float k_f = truncf(phi_total * INV_TWOPI_F);
        const float phi_w = fmaf(-k_f, TWOPI_F, phi_total);
        const float mfr = cosf(phi_w);
        const float mfi = sinf(phi_w);
        acc_r += vr * mfr - vi * mfi;
        acc_i += vr * mfi + vi * mfr;
      }
    }
    __syncthreads();
  }

  if (in_image) {
    simg[2 * image_loc] += acc_r;
    simg[2 * image_loc + 1] += acc_i;
  }
}

#undef HOLOSCAN_RADAR_BP_BLOCK_SIDE
#undef HOLOSCAN_RADAR_BP_BLOCK_PIXELS

}  // namespace holoscan::radar::detail

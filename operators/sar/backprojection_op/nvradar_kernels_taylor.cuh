// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Copied from NVRadar backprojection CUDA kernels
// (modules/edge/python/nvradar/edge/operators/backprojection_image_formation/bp_kernels.py)
// at SHA 37c1cb29355bcfe256ae728f51dd47071847d949. OSRB relicensing of the
// copies is recorded in this operator's OPEN_ITEMS.md. Taylor refined /
// expanded family plus the fp64-trajectory HQ block. Do not copy
// backprojection_kernel (legacy float bin/w).

#pragma once

namespace holoscan::radar::detail {

#define HOLOSCAN_RADAR_BP_TAYLOR_SIDE 16
#define HOLOSCAN_RADAR_BP_TAYLOR_PIXELS \
  (HOLOSCAN_RADAR_BP_TAYLOR_SIDE * HOLOSCAN_RADAR_BP_TAYLOR_SIDE)

// nvradar_block_anchored_refined_quality: Stage-3 Taylor (df-pc +
// df-dR via TwoProduct) + Cody-Waite step-1-only wrap. fp32 dx/ux.
__global__ void
backprojection_kernel_fp64_phiwrap_f32mf_f32vph_anchor_taylor_phi32df_dr_cw1nw(  // NOLINT
    int fcount, int img_width, int img_height, const int Np,
    const int phase_center_stride, const double rmin, const double dr,
    const float dr_inv, const double pc_partial, float* __restrict__ simg,
    const double* __restrict__ fvec, const double* __restrict__ ximg_grid,
    const double* __restrict__ yimg_grid,
    const double* __restrict__ zimg_grid,
    const double* __restrict__ phase_center, const double* __restrict__ R0,
    const float* __restrict__ vph, int Nfft) {
  (void)fcount;
  (void)fvec;
  const double TWOPI_D = 6.283185307179586476925286766559;
  const double INV_TWOPI_D = 0.15915494309189534560822210096757;
  const float INV_TWOPI_F = 0.15915494309189534f;
  const float TWOPI_HI = 6.28125f;
  const float TWOPI_LO = 1.9353071693e-3f;

  __shared__ int s_anchor_int[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_anchor_frac[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_ux[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_uy[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_uz[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_inv_R[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_phi_anchor[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ double s_ax_d;
  __shared__ double s_ay_d;
  __shared__ double s_az_d;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int xi = blockIdx.x * HOLOSCAN_RADAR_BP_TAYLOR_SIDE + threadIdx.x;
  const int yi = blockIdx.y * HOLOSCAN_RADAR_BP_TAYLOR_SIDE + threadIdx.y;
  const bool in_image = (xi < img_width) && (yi < img_height);
  const int image_loc = yi * img_width + xi;

  if (tid == 0) {
    const int ax = blockIdx.x * HOLOSCAN_RADAR_BP_TAYLOR_SIDE +
                   HOLOSCAN_RADAR_BP_TAYLOR_SIDE / 2;
    const int ay = blockIdx.y * HOLOSCAN_RADAR_BP_TAYLOR_SIDE +
                   HOLOSCAN_RADAR_BP_TAYLOR_SIDE / 2;
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

  const float pc_partial_hi = static_cast<float>(pc_partial);
  const float pc_partial_lo =
      static_cast<float>(pc_partial - static_cast<double>(pc_partial_hi));

  float acc_r = 0.0f;
  float acc_i = 0.0f;

  for (int chunk = 0; chunk < Np; chunk += HOLOSCAN_RADAR_BP_TAYLOR_PIXELS) {
    const int remain = Np - chunk;
    const int chunk_size = remain < HOLOSCAN_RADAR_BP_TAYLOR_PIXELS
                               ? remain
                               : HOLOSCAN_RADAR_BP_TAYLOR_PIXELS;

    if (tid < chunk_size) {
      const int p = chunk + tid;
      const double pcx = phase_center[p];
      const double pcy = phase_center[p + phase_center_stride];
      const double pcz = phase_center[p + phase_center_stride * 2];
      const double R0_p_d = R0[p];

      const double dxa = s_ax_d - pcx;
      const double dya = s_ay_d - pcy;
      const double dza = s_az_d - pcz;
      const double R_anchor_pure = sqrt(dxa * dxa + dya * dya + dza * dza);
      const double R_anchor = R_anchor_pure - R0_p_d;
      const double bin_anchor = (R_anchor - rmin) / dr;
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

        // Stage-3 dR: TwoProduct + 2Sum on the dot product, then
        // absorb the small Taylor correction into dR_hi.
        const float t1_hi = ux * dx_a;
        const float t1_lo = fmaf(ux, dx_a, -t1_hi);
        const float t2_hi = uy * dy_a;
        const float t2_lo = fmaf(uy, dy_a, -t2_hi);
        const float t3_hi = uz * dz_a;
        const float t3_lo = fmaf(uz, dz_a, -t3_hi);

        const float s12 = t1_hi + t2_hi;
        const float s12_b = s12 - t1_hi;
        const float e12 = (t1_hi - (s12 - s12_b)) + (t2_hi - s12_b);
        const float dot_hi = s12 + t3_hi;
        const float s_hi_b = dot_hi - s12;
        const float e3 = (s12 - (dot_hi - s_hi_b)) + (t3_hi - s_hi_b);

        const float dot_lo = ((t1_lo + t2_lo) + t3_lo) + (e12 + e3);

        const float dsq =
            fmaf(dx_a, dx_a, fmaf(dy_a, dy_a, dz_a * dz_a));
        const float perp_sq = dsq - dot_hi * dot_hi;
        const float corr =
            perp_sq * 0.5f * invR * fmaf(-dot_hi, invR, 1.0f);
        const float dR_hi = dot_hi + corr;
        const float dR_lo = dot_lo;

        const float dbin = dR_hi * dr_inv;
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

        // Stage-3 phase: full bilinear extension of
        // (pc_hi, pc_lo) * (dR_hi, dR_lo).
        const float p_hi = pc_partial_hi * dR_hi;
        const float p_err = fmaf(pc_partial_hi, dR_hi, -p_hi);
        const float p_cross = pc_partial_hi * dR_lo;
        const float p_corr = pc_partial_lo * dR_hi;
        const float p_lo = (p_err + p_cross) + p_corr;

        // CW wrap step 1 only; libm cosf/sinf reduce arguments
        // in (-2π, 2π) internally.
        const float k1 = truncf(p_hi * INV_TWOPI_F);
        const float w1 = fmaf(-k1, TWOPI_HI, p_hi);
        const float p_hi_w = fmaf(-k1, TWOPI_LO, w1);

        const float phi_t = (s_phi_anchor[j] + p_hi_w) + p_lo;
        const float mfr = cosf(phi_t);
        const float mfi = sinf(phi_t);

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

// nvradar_block_anchored_expanded_throughput: Stage-5c + double-float
// dx/ux + FMA-folded inner loop.
__global__ void
backprojection_kernel_fp64_phiwrap_f32mf_f32vph_anchor_taylor_phi32df_dr_in_cw1nw_fma(  // NOLINT
    int fcount, int img_width, int img_height, const int Np,
    const int phase_center_stride, const double rmin, const double dr,
    const float dr_inv, const double pc_partial, float* __restrict__ simg,
    const double* __restrict__ fvec, const double* __restrict__ ximg_grid,
    const double* __restrict__ yimg_grid,
    const double* __restrict__ zimg_grid,
    const double* __restrict__ phase_center, const double* __restrict__ R0,
    const float* __restrict__ vph, int Nfft) {
  (void)fcount;
  (void)fvec;
  const double TWOPI_D = 6.283185307179586476925286766559;
  const double INV_TWOPI_D = 0.15915494309189534560822210096757;
  const float INV_TWOPI_F = 0.15915494309189534f;
  const float TWOPI_HI = 6.28125f;
  const float TWOPI_LO = 1.9353071693e-3f;

  __shared__ int s_anchor_int[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_anchor_frac[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_ux_hi[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_uy_hi[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_uz_hi[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_ux_lo[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_uy_lo[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_uz_lo[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_inv_R[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_phi_anchor[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ double s_ax_d;
  __shared__ double s_ay_d;
  __shared__ double s_az_d;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int xi = blockIdx.x * HOLOSCAN_RADAR_BP_TAYLOR_SIDE + threadIdx.x;
  const int yi = blockIdx.y * HOLOSCAN_RADAR_BP_TAYLOR_SIDE + threadIdx.y;
  const bool in_image = (xi < img_width) && (yi < img_height);
  const int image_loc = yi * img_width + xi;

  if (tid == 0) {
    const int ax = blockIdx.x * HOLOSCAN_RADAR_BP_TAYLOR_SIDE +
                   HOLOSCAN_RADAR_BP_TAYLOR_SIDE / 2;
    const int ay = blockIdx.y * HOLOSCAN_RADAR_BP_TAYLOR_SIDE +
                   HOLOSCAN_RADAR_BP_TAYLOR_SIDE / 2;
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
  const double dx_full_d = x_d - s_ax_d;
  const double dy_full_d = y_d - s_ay_d;
  const double dz_full_d = z_d - s_az_d;
  const float dx_hi = in_image ? static_cast<float>(dx_full_d) : 0.0f;
  const float dy_hi = in_image ? static_cast<float>(dy_full_d) : 0.0f;
  const float dz_hi = in_image ? static_cast<float>(dz_full_d) : 0.0f;
  const float dx_lo = in_image
                          ? static_cast<float>(dx_full_d -
                                               static_cast<double>(dx_hi))
                          : 0.0f;
  const float dy_lo = in_image
                          ? static_cast<float>(dy_full_d -
                                               static_cast<double>(dy_hi))
                          : 0.0f;
  const float dz_lo = in_image
                          ? static_cast<float>(dz_full_d -
                                               static_cast<double>(dz_hi))
                          : 0.0f;

  const float pc_partial_hi = static_cast<float>(pc_partial);
  const float pc_partial_lo =
      static_cast<float>(pc_partial - static_cast<double>(pc_partial_hi));

  float acc_r = 0.0f;
  float acc_i = 0.0f;

  for (int chunk = 0; chunk < Np; chunk += HOLOSCAN_RADAR_BP_TAYLOR_PIXELS) {
    const int remain = Np - chunk;
    const int chunk_size = remain < HOLOSCAN_RADAR_BP_TAYLOR_PIXELS
                               ? remain
                               : HOLOSCAN_RADAR_BP_TAYLOR_PIXELS;

    if (tid < chunk_size) {
      const int p = chunk + tid;
      const double pcx = phase_center[p];
      const double pcy = phase_center[p + phase_center_stride];
      const double pcz = phase_center[p + phase_center_stride * 2];
      const double R0_p_d = R0[p];

      const double dxa = s_ax_d - pcx;
      const double dya = s_ay_d - pcy;
      const double dza = s_az_d - pcz;
      const double R_anchor_pure = sqrt(dxa * dxa + dya * dya + dza * dza);
      const double R_anchor = R_anchor_pure - R0_p_d;
      const double bin_anchor = (R_anchor - rmin) / dr;
      const double bin_floor = floor(bin_anchor);
      const double inv_R_d = 1.0 / R_anchor_pure;

      const double ux_full = dxa * inv_R_d;
      const double uy_full = dya * inv_R_d;
      const double uz_full = dza * inv_R_d;
      const float ux_hi = static_cast<float>(ux_full);
      const float uy_hi = static_cast<float>(uy_full);
      const float uz_hi = static_cast<float>(uz_full);

      const double phi_anchor_d = pc_partial * R_anchor;
      const double k_a = trunc(phi_anchor_d * INV_TWOPI_D);
      const double phi_anchor_w = fma(-k_a, TWOPI_D, phi_anchor_d);

      s_anchor_int[tid] = static_cast<int>(bin_floor);
      s_anchor_frac[tid] = static_cast<float>(bin_anchor - bin_floor);
      s_ux_hi[tid] = ux_hi;
      s_uy_hi[tid] = uy_hi;
      s_uz_hi[tid] = uz_hi;
      s_ux_lo[tid] = static_cast<float>(ux_full - static_cast<double>(ux_hi));
      s_uy_lo[tid] = static_cast<float>(uy_full - static_cast<double>(uy_hi));
      s_uz_lo[tid] = static_cast<float>(uz_full - static_cast<double>(uz_hi));
      s_inv_R[tid] = static_cast<float>(inv_R_d);
      s_phi_anchor[tid] = static_cast<float>(phi_anchor_w);
    }
    __syncthreads();

    if (in_image) {
#pragma unroll 4
      for (int j = 0; j < chunk_size; ++j) {
        const float ux_hi = s_ux_hi[j];
        const float uy_hi = s_uy_hi[j];
        const float uz_hi = s_uz_hi[j];
        const float ux_lo = s_ux_lo[j];
        const float uy_lo = s_uy_lo[j];
        const float uz_lo = s_uz_lo[j];
        const float invR = s_inv_R[j];

        const float t1_hi = ux_hi * dx_hi;
        const float t1_lo = fmaf(ux_hi, dx_hi, -t1_hi);
        const float t1_off = fmaf(ux_lo, dx_hi, ux_hi * dx_lo);
        const float t2_hi = uy_hi * dy_hi;
        const float t2_lo = fmaf(uy_hi, dy_hi, -t2_hi);
        const float t2_off = fmaf(uy_lo, dy_hi, uy_hi * dy_lo);
        const float t3_hi = uz_hi * dz_hi;
        const float t3_lo = fmaf(uz_hi, dz_hi, -t3_hi);
        const float t3_off = fmaf(uz_lo, dz_hi, uz_hi * dz_lo);

        const float s12 = t1_hi + t2_hi;
        const float s12_b = s12 - t1_hi;
        const float e12 = (t1_hi - (s12 - s12_b)) + (t2_hi - s12_b);
        const float dot_hi = s12 + t3_hi;
        const float s_hi_b = dot_hi - s12;
        const float e3 = (s12 - (dot_hi - s_hi_b)) + (t3_hi - s_hi_b);

        const float dot_lo = ((t1_lo + t2_lo) + t3_lo) +
                             ((t1_off + t2_off) + t3_off) + (e12 + e3);

        // perp_sq: explicit fmaf to fold the dsq - dot_hi*dot_hi
        // cancellation into a single rounding step.
        const float dsq =
            fmaf(dx_hi, dx_hi, fmaf(dy_hi, dy_hi, dz_hi * dz_hi));
        const float perp_sq = fmaf(-dot_hi, dot_hi, dsq);
        const float corr =
            perp_sq * 0.5f * invR * fmaf(-dot_hi, invR, 1.0f);

        const float dR_hi = dot_hi + corr;
        const float dR_corr_err = corr - (dR_hi - dot_hi);
        const float dR_lo = dot_lo + dR_corr_err;

        // bin_loc: fold dR_hi*dr_inv + frac into one fmaf.
        const float bin_loc = fmaf(dR_hi, dr_inv, s_anchor_frac[j]);
        const float bf_floor = floorf(bin_loc);
        const int b1 = s_anchor_int[j] + static_cast<int>(bf_floor);
        const float w = bin_loc - bf_floor;
        if ((b1 < 0) || (b1 >= Nfft - 1)) {
          continue;
        }

        // vph linear interp: fold v1 + w*(v2 - v1) into one fmaf.
        const int p = chunk + j;
        const int pulse_offset = p * Nfft * 2;
        const float v1r = vph[pulse_offset + b1 * 2];
        const float v1i = vph[pulse_offset + b1 * 2 + 1];
        const float v2r = vph[pulse_offset + b1 * 2 + 2];
        const float v2i = vph[pulse_offset + b1 * 2 + 3];
        const float vr = fmaf(w, v2r - v1r, v1r);
        const float vi = fmaf(w, v2i - v1i, v1i);

        // p_hi / p_lo: collapse (p_err + p_cross) + p_corr into
        // a single accumulator updated by chained fmafs.
        const float p_hi = pc_partial_hi * dR_hi;
        float p_lo = fmaf(pc_partial_hi, dR_hi, -p_hi);
        p_lo = fmaf(pc_partial_hi, dR_lo, p_lo);
        p_lo = fmaf(pc_partial_lo, dR_hi, p_lo);

        const float k1 = truncf(p_hi * INV_TWOPI_F);
        const float w1 = fmaf(-k1, TWOPI_HI, p_hi);
        const float p_hi_w = fmaf(-k1, TWOPI_LO, w1);

        const float phi_t = (s_phi_anchor[j] + p_hi_w) + p_lo;
        const float mfr = cosf(phi_t);
        const float mfi = sinf(phi_t);

        acc_r = fmaf(vr, mfr, fmaf(-vi, mfi, acc_r));
        acc_i = fmaf(vr, mfi, fmaf(vi, mfr, acc_i));
      }
    }
    __syncthreads();
  }

  if (in_image) {
    simg[2 * image_loc] += acc_r;
    simg[2 * image_loc + 1] += acc_i;
  }
}

// nvradar_block_anchored_expanded_quality: Stage-5c + double-float
// dx/ux, non-FMA inner loop.
__global__ void
backprojection_kernel_fp64_phiwrap_f32mf_f32vph_anchor_taylor_phi32df_dr_in_cw1nw(  // NOLINT
    int fcount, int img_width, int img_height, const int Np,
    const int phase_center_stride, const double rmin, const double dr,
    const float dr_inv, const double pc_partial, float* __restrict__ simg,
    const double* __restrict__ fvec, const double* __restrict__ ximg_grid,
    const double* __restrict__ yimg_grid,
    const double* __restrict__ zimg_grid,
    const double* __restrict__ phase_center, const double* __restrict__ R0,
    const float* __restrict__ vph, int Nfft) {
  (void)fcount;
  (void)fvec;
  const double TWOPI_D = 6.283185307179586476925286766559;
  const double INV_TWOPI_D = 0.15915494309189534560822210096757;
  const float INV_TWOPI_F = 0.15915494309189534f;
  const float TWOPI_HI = 6.28125f;
  const float TWOPI_LO = 1.9353071693e-3f;

  __shared__ int s_anchor_int[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_anchor_frac[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_ux_hi[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_uy_hi[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_uz_hi[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_ux_lo[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_uy_lo[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_uz_lo[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_inv_R[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_phi_anchor[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ double s_ax_d;
  __shared__ double s_ay_d;
  __shared__ double s_az_d;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int xi = blockIdx.x * HOLOSCAN_RADAR_BP_TAYLOR_SIDE + threadIdx.x;
  const int yi = blockIdx.y * HOLOSCAN_RADAR_BP_TAYLOR_SIDE + threadIdx.y;
  const bool in_image = (xi < img_width) && (yi < img_height);
  const int image_loc = yi * img_width + xi;

  if (tid == 0) {
    const int ax = blockIdx.x * HOLOSCAN_RADAR_BP_TAYLOR_SIDE +
                   HOLOSCAN_RADAR_BP_TAYLOR_SIDE / 2;
    const int ay = blockIdx.y * HOLOSCAN_RADAR_BP_TAYLOR_SIDE +
                   HOLOSCAN_RADAR_BP_TAYLOR_SIDE / 2;
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
  const double dx_full_d = x_d - s_ax_d;
  const double dy_full_d = y_d - s_ay_d;
  const double dz_full_d = z_d - s_az_d;
  const float dx_hi = in_image ? static_cast<float>(dx_full_d) : 0.0f;
  const float dy_hi = in_image ? static_cast<float>(dy_full_d) : 0.0f;
  const float dz_hi = in_image ? static_cast<float>(dz_full_d) : 0.0f;
  const float dx_lo = in_image
                          ? static_cast<float>(dx_full_d -
                                               static_cast<double>(dx_hi))
                          : 0.0f;
  const float dy_lo = in_image
                          ? static_cast<float>(dy_full_d -
                                               static_cast<double>(dy_hi))
                          : 0.0f;
  const float dz_lo = in_image
                          ? static_cast<float>(dz_full_d -
                                               static_cast<double>(dz_hi))
                          : 0.0f;

  const float pc_partial_hi = static_cast<float>(pc_partial);
  const float pc_partial_lo =
      static_cast<float>(pc_partial - static_cast<double>(pc_partial_hi));

  float acc_r = 0.0f;
  float acc_i = 0.0f;

  for (int chunk = 0; chunk < Np; chunk += HOLOSCAN_RADAR_BP_TAYLOR_PIXELS) {
    const int remain = Np - chunk;
    const int chunk_size = remain < HOLOSCAN_RADAR_BP_TAYLOR_PIXELS
                               ? remain
                               : HOLOSCAN_RADAR_BP_TAYLOR_PIXELS;

    if (tid < chunk_size) {
      const int p = chunk + tid;
      const double pcx = phase_center[p];
      const double pcy = phase_center[p + phase_center_stride];
      const double pcz = phase_center[p + phase_center_stride * 2];
      const double R0_p_d = R0[p];

      const double dxa = s_ax_d - pcx;
      const double dya = s_ay_d - pcy;
      const double dza = s_az_d - pcz;
      const double R_anchor_pure = sqrt(dxa * dxa + dya * dya + dza * dza);
      const double R_anchor = R_anchor_pure - R0_p_d;
      const double bin_anchor = (R_anchor - rmin) / dr;
      const double bin_floor = floor(bin_anchor);
      const double inv_R_d = 1.0 / R_anchor_pure;

      const double ux_full = dxa * inv_R_d;
      const double uy_full = dya * inv_R_d;
      const double uz_full = dza * inv_R_d;
      const float ux_hi = static_cast<float>(ux_full);
      const float uy_hi = static_cast<float>(uy_full);
      const float uz_hi = static_cast<float>(uz_full);

      const double phi_anchor_d = pc_partial * R_anchor;
      const double k_a = trunc(phi_anchor_d * INV_TWOPI_D);
      const double phi_anchor_w = fma(-k_a, TWOPI_D, phi_anchor_d);

      s_anchor_int[tid] = static_cast<int>(bin_floor);
      s_anchor_frac[tid] = static_cast<float>(bin_anchor - bin_floor);
      s_ux_hi[tid] = ux_hi;
      s_uy_hi[tid] = uy_hi;
      s_uz_hi[tid] = uz_hi;
      s_ux_lo[tid] = static_cast<float>(ux_full - static_cast<double>(ux_hi));
      s_uy_lo[tid] = static_cast<float>(uy_full - static_cast<double>(uy_hi));
      s_uz_lo[tid] = static_cast<float>(uz_full - static_cast<double>(uz_hi));
      s_inv_R[tid] = static_cast<float>(inv_R_d);
      s_phi_anchor[tid] = static_cast<float>(phi_anchor_w);
    }
    __syncthreads();

    if (in_image) {
#pragma unroll 4
      for (int j = 0; j < chunk_size; ++j) {
        const float ux_hi = s_ux_hi[j];
        const float uy_hi = s_uy_hi[j];
        const float uz_hi = s_uz_hi[j];
        const float ux_lo = s_ux_lo[j];
        const float uy_lo = s_uy_lo[j];
        const float uz_lo = s_uz_lo[j];
        const float invR = s_inv_R[j];

        const float t1_hi = ux_hi * dx_hi;
        const float t1_lo = fmaf(ux_hi, dx_hi, -t1_hi);
        const float t1_off = fmaf(ux_lo, dx_hi, ux_hi * dx_lo);
        const float t2_hi = uy_hi * dy_hi;
        const float t2_lo = fmaf(uy_hi, dy_hi, -t2_hi);
        const float t2_off = fmaf(uy_lo, dy_hi, uy_hi * dy_lo);
        const float t3_hi = uz_hi * dz_hi;
        const float t3_lo = fmaf(uz_hi, dz_hi, -t3_hi);
        const float t3_off = fmaf(uz_lo, dz_hi, uz_hi * dz_lo);

        const float s12 = t1_hi + t2_hi;
        const float s12_b = s12 - t1_hi;
        const float e12 = (t1_hi - (s12 - s12_b)) + (t2_hi - s12_b);
        const float dot_hi = s12 + t3_hi;
        const float s_hi_b = dot_hi - s12;
        const float e3 = (s12 - (dot_hi - s_hi_b)) + (t3_hi - s_hi_b);

        const float dot_lo = ((t1_lo + t2_lo) + t3_lo) +
                             ((t1_off + t2_off) + t3_off) + (e12 + e3);

        const float dsq =
            fmaf(dx_hi, dx_hi, fmaf(dy_hi, dy_hi, dz_hi * dz_hi));
        const float perp_sq = dsq - dot_hi * dot_hi;
        const float corr =
            perp_sq * 0.5f * invR * fmaf(-dot_hi, invR, 1.0f);

        const float dR_hi = dot_hi + corr;
        const float dR_corr_err = corr - (dR_hi - dot_hi);
        const float dR_lo = dot_lo + dR_corr_err;

        const float dbin = dR_hi * dr_inv;
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

        const float p_hi = pc_partial_hi * dR_hi;
        const float p_err = fmaf(pc_partial_hi, dR_hi, -p_hi);
        const float p_cross = pc_partial_hi * dR_lo;
        const float p_corr = pc_partial_lo * dR_hi;
        const float p_lo = (p_err + p_cross) + p_corr;

        // CW wrap on step 1, then feed phi_t directly to libm
        // (cosf/sinf reduce internally; phi_t ∈ (-2π, 2π)).
        const float k1 = truncf(p_hi * INV_TWOPI_F);
        const float w1 = fmaf(-k1, TWOPI_HI, p_hi);
        const float p_hi_w = fmaf(-k1, TWOPI_LO, w1);

        const float phi_t = (s_phi_anchor[j] + p_hi_w) + p_lo;
        const float mfr = cosf(phi_t);
        const float mfi = sinf(phi_t);

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

// nvradar_block_anchored_fp64_trajectory: HQ identity at a per-block
// fp32-grid anchor, fp64 trajectory / mcp / R0, 3rd-order Taylor dR.
__global__ void backprojection_kernel_fp32_hq_block3_fp64in(
    int fcount, int img_width, int img_height, const int Np,
    const int phase_center_stride, const int mcp_stride, const float rmin,
    const float dr_inv, const float pc_partial, float* __restrict__ simg,
    const float* __restrict__ fvec, const float* __restrict__ ximg_grid,
    const float* __restrict__ yimg_grid, const float* __restrict__ zimg_grid,
    const double* __restrict__ phase_center, const double* __restrict__ mcp,
    const double* __restrict__ R0, const float* __restrict__ vph, int Nfft) {
  (void)fcount;
  (void)fvec;
  const float INV_TWOPI_F = 0.15915494309189534f;
  const float TWOPI_F = 6.2831853071795865f;
  const double INV_TWOPI_D = 0.15915494309189534560822210096757;
  const double TWOPI_D = 6.283185307179586476925286766559;

  __shared__ float s_phi[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_ux[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_uy[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_uz[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_invR[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_R_rel[HOLOSCAN_RADAR_BP_TAYLOR_PIXELS];
  __shared__ float s_ax;
  __shared__ float s_ay;
  __shared__ float s_az;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int xi = blockIdx.x * HOLOSCAN_RADAR_BP_TAYLOR_SIDE + threadIdx.x;
  const int yi = blockIdx.y * HOLOSCAN_RADAR_BP_TAYLOR_SIDE + threadIdx.y;
  const bool in_image = (xi < img_width) && (yi < img_height);
  const int image_loc = yi * img_width + xi;

  if (tid == 0) {
    const int ax = blockIdx.x * HOLOSCAN_RADAR_BP_TAYLOR_SIDE +
                   HOLOSCAN_RADAR_BP_TAYLOR_SIDE / 2;
    const int ay = blockIdx.y * HOLOSCAN_RADAR_BP_TAYLOR_SIDE +
                   HOLOSCAN_RADAR_BP_TAYLOR_SIDE / 2;
    const int ax_c = ax < img_width ? ax : (img_width - 1);
    const int ay_c = ay < img_height ? ay : (img_height - 1);
    const int anchor_idx = ay_c * img_width + ax_c;
    s_ax = ximg_grid[anchor_idx];
    s_ay = yimg_grid[anchor_idx];
    s_az = zimg_grid[anchor_idx];
  }
  __syncthreads();

  const float x = in_image ? ximg_grid[image_loc] : 0.0f;
  const float y = in_image ? yimg_grid[image_loc] : 0.0f;
  const float z = in_image ? zimg_grid[image_loc] : 0.0f;
  const float dx_a = x - s_ax;
  const float dy_a = y - s_ay;
  const float dz_a = z - s_az;

  float acc_r = 0.0f;
  float acc_i = 0.0f;

  for (int chunk = 0; chunk < Np; chunk += HOLOSCAN_RADAR_BP_TAYLOR_PIXELS) {
    const int remain = Np - chunk;
    const int chunk_size = remain < HOLOSCAN_RADAR_BP_TAYLOR_PIXELS
                               ? remain
                               : HOLOSCAN_RADAR_BP_TAYLOR_PIXELS;

    if (tid < chunk_size) {
      const int p = chunk + tid;
      const double ax_d = static_cast<double>(s_ax);
      const double ay_d = static_cast<double>(s_ay);
      const double az_d = static_cast<double>(s_az);
      // Native fp64 loads: no precision-discarding cast.
      const double pcx = phase_center[p];
      const double pcy = phase_center[p + phase_center_stride];
      const double pcz = phase_center[p + phase_center_stride * 2];
      const double mx = mcp[p];
      const double my = mcp[p + mcp_stride];
      const double mz = mcp[p + mcp_stride * 2];

      const double dx = ax_d - mx;
      const double dy = ay_d - my;
      const double dz = az_d - mz;
      const double vx = mx - pcx;
      const double vy = my - pcy;
      const double vz = mz - pcz;
      const double d_sq = dx * dx + dy * dy + dz * dz;
      const double v_dot_d = vx * dx + vy * dy + vz * dz;
      const double numer = d_sq + 2.0 * v_dot_d;
      const double R0_p = R0[p];
      const double R_raw = sqrt(numer + R0_p * R0_p);
      const double R_rel = numer / (R_raw + R0_p);

      const double phase = static_cast<double>(pc_partial) * R_rel;
      const double k = trunc(phase * INV_TWOPI_D);
      const double phi_red = fma(-k, TWOPI_D, phase);

      const double dxp = ax_d - pcx;
      const double dyp = ay_d - pcy;
      const double dzp = az_d - pcz;
      const double R_anchor = sqrt(dxp * dxp + dyp * dyp + dzp * dzp);
      const double inv_R = 1.0 / R_anchor;

      s_phi[tid] = static_cast<float>(phi_red);
      s_ux[tid] = static_cast<float>(dxp * inv_R);
      s_uy[tid] = static_cast<float>(dyp * inv_R);
      s_uz[tid] = static_cast<float>(dzp * inv_R);
      s_invR[tid] = static_cast<float>(inv_R);
      s_R_rel[tid] = static_cast<float>(R_rel);
    }
    __syncthreads();

    if (in_image) {
#pragma unroll 4
      for (int j = 0; j < chunk_size; ++j) {
        const float ux = s_ux[j];
        const float uy = s_uy[j];
        const float uz = s_uz[j];
        const float inv_R = s_invR[j];
        const float dot = dx_a * ux + dy_a * uy + dz_a * uz;
        const float dsq = dx_a * dx_a + dy_a * dy_a + dz_a * dz_a;
        const float perp_sq = dsq - dot * dot;
        const float dR =
            dot + perp_sq * 0.5f * inv_R * (1.0f - dot * inv_R);

        const float R_rel_pixel = s_R_rel[j] + dR;
        const float bin = (R_rel_pixel - rmin) * dr_inv;
        if ((bin < 0.0f) || (bin >= static_cast<float>(Nfft - 1))) {
          continue;
        }
        const int b1 = static_cast<int>(floorf(bin));
        const float w = bin - floorf(bin);

        float dphi = pc_partial * dR;
        dphi -= TWOPI_F * truncf(dphi * INV_TWOPI_F);
        float phi = s_phi[j] + dphi;
        phi -= TWOPI_F * truncf(phi * INV_TWOPI_F);

        float mfr = 0.0f;
        float mfi = 0.0f;
        sincosf(phi, &mfi, &mfr);

        const int p = chunk + j;
        const int pulse_offset = p * Nfft * 2;
        const float v1r = vph[pulse_offset + b1 * 2];
        const float v1i = vph[pulse_offset + b1 * 2 + 1];
        const float v2r = vph[pulse_offset + b1 * 2 + 2];
        const float v2i = vph[pulse_offset + b1 * 2 + 3];
        const float vr = v1r + w * (v2r - v1r);
        const float vi = v1i + w * (v2i - v1i);
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

#undef HOLOSCAN_RADAR_BP_TAYLOR_SIDE
#undef HOLOSCAN_RADAR_BP_TAYLOR_PIXELS

}  // namespace holoscan::radar::detail

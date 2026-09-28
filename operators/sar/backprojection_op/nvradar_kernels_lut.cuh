// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Copied from NVRadar backprojection CUDA kernels
// (modules/edge/python/nvradar/edge/operators/backprojection_image_formation/bp_kernels.py)
// at SHA 37c1cb29355bcfe256ae728f51dd47071847d949. OSRB relicensing of the
// copies is recorded in this operator's OPEN_ITEMS.md. Phase-LUT family
// only. Do not copy backprojection_kernel (legacy float bin/w). Do not
// share a phase LUT with MatX — the fills differ.

#pragma once

namespace holoscan::radar::detail {

#define HOLOSCAN_RADAR_BP_LUT_SIDE 16
#define HOLOSCAN_RADAR_BP_LUT_PIXELS (HOLOSCAN_RADAR_BP_LUT_SIDE * HOLOSCAN_RADAR_BP_LUT_SIDE)

// NVRadar build_phase_lut (bp_prepare.py): at integer bin b,
//   phi = pc_partial * (rmin + b * dr)   (fp64)
//   lut[2*b]     = (float)cos(phi)
//   lut[2*b + 1] = (float)sin(phi)
// Interleaved complex64. Distinct from MatX's LUT fill.
// Launch as 1-D: one thread per bin.
__host__ __device__ inline void nvradar_fill_phase_lut_bin(float* lut, int b, double pc_partial,
                                                           double rmin, double dr) {
  const double phi = (static_cast<double>(b) * dr + rmin) * pc_partial;
  lut[2 * b] = static_cast<float>(cos(phi));
  lut[2 * b + 1] = static_cast<float>(sin(phi));
}

__global__ void nvradar_fill_phase_lut(float* lut, double pc_partial, double rmin, double dr,
                                       int Nfft) {
  const int b = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (b >= Nfft) {
    return;
  }
  nvradar_fill_phase_lut_bin(lut, b, pc_partial, rmin, dr);
}

// nvradar_block_anchored_phase_lut: Taylor refined-throughput + LUT
// phase. Takes (rmin, dr_inv_d, dr_inv, pc_partial_dr).
__global__ void backprojection_kernel_anchor_taylor_phase_lut(
    int fcount, int img_width, int img_height, const int Np, const int phase_center_stride,
    const double rmin, const double dr_inv_d, const float dr_inv, const float pc_partial_dr,
    float* __restrict__ simg, const double* __restrict__ fvec, const double* __restrict__ ximg_grid,
    const double* __restrict__ yimg_grid, const double* __restrict__ zimg_grid,
    const double* __restrict__ phase_center, const double* __restrict__ R0,
    const float* __restrict__ vph, const float* __restrict__ phase_lut, int Nfft) {
  (void)fcount;
  (void)fvec;
  __shared__ int s_anchor_int[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_anchor_frac[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_ux[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_uy[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_uz[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_inv_R[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ double s_ax_d;
  __shared__ double s_ay_d;
  __shared__ double s_az_d;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int xi = blockIdx.x * HOLOSCAN_RADAR_BP_LUT_SIDE + threadIdx.x;
  const int yi = blockIdx.y * HOLOSCAN_RADAR_BP_LUT_SIDE + threadIdx.y;
  const bool in_image = (xi < img_width) && (yi < img_height);
  const int image_loc = yi * img_width + xi;

  if (tid == 0) {
    const int ax = blockIdx.x * HOLOSCAN_RADAR_BP_LUT_SIDE + HOLOSCAN_RADAR_BP_LUT_SIDE / 2;
    const int ay = blockIdx.y * HOLOSCAN_RADAR_BP_LUT_SIDE + HOLOSCAN_RADAR_BP_LUT_SIDE / 2;
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

  float acc_r = 0.0f;
  float acc_i = 0.0f;

  for (int chunk = 0; chunk < Np; chunk += HOLOSCAN_RADAR_BP_LUT_PIXELS) {
    const int remain = Np - chunk;
    const int chunk_size =
        remain < HOLOSCAN_RADAR_BP_LUT_PIXELS ? remain : HOLOSCAN_RADAR_BP_LUT_PIXELS;

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
      const double bin_anchor = (R_anchor - rmin) * dr_inv_d;
      const double bin_floor = floor(bin_anchor);
      const double inv_R_d = 1.0 / R_anchor_pure;

      s_anchor_int[tid] = static_cast<int>(bin_floor);
      s_anchor_frac[tid] = static_cast<float>(bin_anchor - bin_floor);
      s_ux[tid] = static_cast<float>(dxa * inv_R_d);
      s_uy[tid] = static_cast<float>(dya * inv_R_d);
      s_uz[tid] = static_cast<float>(dza * inv_R_d);
      s_inv_R[tid] = static_cast<float>(inv_R_d);
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
        const float dR = dot + perp_sq * 0.5f * invR * (1.0f - dot * invR);

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

        // Phase from LUT: complex (cos,sin) at integer bin, rotated
        // by the fractional-bin phase correction via complex multiply.
        const float base_r = phase_lut[2 * b1];
        const float base_i = phase_lut[2 * b1 + 1];
        float incr_sin = 0.0f;
        float incr_cos = 0.0f;
        __sincosf(pc_partial_dr * w, &incr_sin, &incr_cos);
        const float mfr = base_r * incr_cos - base_i * incr_sin;
        const float mfi = base_r * incr_sin + base_i * incr_cos;

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

// nvradar_block_anchored_refined_quality_phase_lut: TwoProduct dR_hi
// (parent's dR_lo trail dropped), LUT + __sincosf.
__global__ void backprojection_kernel_anchor_taylor_quality_phase_lut(
    int fcount, int img_width, int img_height, const int Np, const int phase_center_stride,
    const double rmin, const double dr, const float dr_inv, const float pc_partial_dr,
    float* __restrict__ simg, const double* __restrict__ fvec, const double* __restrict__ ximg_grid,
    const double* __restrict__ yimg_grid, const double* __restrict__ zimg_grid,
    const double* __restrict__ phase_center, const double* __restrict__ R0,
    const float* __restrict__ vph, const float* __restrict__ phase_lut, int Nfft) {
  (void)fcount;
  (void)fvec;
  __shared__ int s_anchor_int[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_anchor_frac[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_ux[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_uy[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_uz[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_inv_R[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ double s_ax_d;
  __shared__ double s_ay_d;
  __shared__ double s_az_d;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int xi = blockIdx.x * HOLOSCAN_RADAR_BP_LUT_SIDE + threadIdx.x;
  const int yi = blockIdx.y * HOLOSCAN_RADAR_BP_LUT_SIDE + threadIdx.y;
  const bool in_image = (xi < img_width) && (yi < img_height);
  const int image_loc = yi * img_width + xi;

  if (tid == 0) {
    const int ax = blockIdx.x * HOLOSCAN_RADAR_BP_LUT_SIDE + HOLOSCAN_RADAR_BP_LUT_SIDE / 2;
    const int ay = blockIdx.y * HOLOSCAN_RADAR_BP_LUT_SIDE + HOLOSCAN_RADAR_BP_LUT_SIDE / 2;
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

  float acc_r = 0.0f;
  float acc_i = 0.0f;

  for (int chunk = 0; chunk < Np; chunk += HOLOSCAN_RADAR_BP_LUT_PIXELS) {
    const int remain = Np - chunk;
    const int chunk_size =
        remain < HOLOSCAN_RADAR_BP_LUT_PIXELS ? remain : HOLOSCAN_RADAR_BP_LUT_PIXELS;

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

      s_anchor_int[tid] = static_cast<int>(bin_floor);
      s_anchor_frac[tid] = static_cast<float>(bin_anchor - bin_floor);
      s_ux[tid] = static_cast<float>(dxa * inv_R_d);
      s_uy[tid] = static_cast<float>(dya * inv_R_d);
      s_uz[tid] = static_cast<float>(dza * inv_R_d);
      s_inv_R[tid] = static_cast<float>(inv_R_d);
    }
    __syncthreads();

    if (in_image) {
#pragma unroll 4
      for (int j = 0; j < chunk_size; ++j) {
        const float ux = s_ux[j];
        const float uy = s_uy[j];
        const float uz = s_uz[j];
        const float invR = s_inv_R[j];

        // Phase comes from the LUT below, so we only need dR_hi;
        // the TwoProduct / 2Sum trail used by the parent kernel to
        // build dR_lo for the bilinear phase product is dropped.
        const float t1_hi = ux * dx_a;
        const float t2_hi = uy * dy_a;
        const float t3_hi = uz * dz_a;
        const float s12 = t1_hi + t2_hi;
        const float dot_hi = s12 + t3_hi;

        const float dsq = fmaf(dx_a, dx_a, fmaf(dy_a, dy_a, dz_a * dz_a));
        const float perp_sq = dsq - dot_hi * dot_hi;
        const float corr = perp_sq * 0.5f * invR * fmaf(-dot_hi, invR, 1.0f);
        const float dR_hi = dot_hi + corr;

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

        const float base_r = phase_lut[2 * b1];
        const float base_i = phase_lut[2 * b1 + 1];
        float incr_sin = 0.0f;
        float incr_cos = 0.0f;
        __sincosf(pc_partial_dr * w, &incr_sin, &incr_cos);
        const float mfr = base_r * incr_cos - base_i * incr_sin;
        const float mfi = base_r * incr_sin + base_i * incr_cos;

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

// nvradar_block_anchored_expanded_throughput_phase_lut: FMA dR + LUT.
__global__ void backprojection_kernel_anchor_expanded_throughput_phase_lut(
    int fcount, int img_width, int img_height, const int Np, const int phase_center_stride,
    const double rmin, const double dr, const float dr_inv, const float pc_partial_dr,
    float* __restrict__ simg, const double* __restrict__ fvec, const double* __restrict__ ximg_grid,
    const double* __restrict__ yimg_grid, const double* __restrict__ zimg_grid,
    const double* __restrict__ phase_center, const double* __restrict__ R0,
    const float* __restrict__ vph, const float* __restrict__ phase_lut, int Nfft) {
  (void)fcount;
  (void)fvec;
  // Phase comes from the LUT so we drop the parent's double-float dx/u
  // decomposition: the _lo shared arrays and the cross-term machinery
  // feeding dot_lo / dR_lo have no consumer in this variant.
  __shared__ int s_anchor_int[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_anchor_frac[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_ux_hi[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_uy_hi[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_uz_hi[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_inv_R[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ double s_ax_d;
  __shared__ double s_ay_d;
  __shared__ double s_az_d;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int xi = blockIdx.x * HOLOSCAN_RADAR_BP_LUT_SIDE + threadIdx.x;
  const int yi = blockIdx.y * HOLOSCAN_RADAR_BP_LUT_SIDE + threadIdx.y;
  const bool in_image = (xi < img_width) && (yi < img_height);
  const int image_loc = yi * img_width + xi;

  if (tid == 0) {
    const int ax = blockIdx.x * HOLOSCAN_RADAR_BP_LUT_SIDE + HOLOSCAN_RADAR_BP_LUT_SIDE / 2;
    const int ay = blockIdx.y * HOLOSCAN_RADAR_BP_LUT_SIDE + HOLOSCAN_RADAR_BP_LUT_SIDE / 2;
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
  const float dx_hi = in_image ? static_cast<float>(x_d - s_ax_d) : 0.0f;
  const float dy_hi = in_image ? static_cast<float>(y_d - s_ay_d) : 0.0f;
  const float dz_hi = in_image ? static_cast<float>(z_d - s_az_d) : 0.0f;

  float acc_r = 0.0f;
  float acc_i = 0.0f;

  for (int chunk = 0; chunk < Np; chunk += HOLOSCAN_RADAR_BP_LUT_PIXELS) {
    const int remain = Np - chunk;
    const int chunk_size =
        remain < HOLOSCAN_RADAR_BP_LUT_PIXELS ? remain : HOLOSCAN_RADAR_BP_LUT_PIXELS;

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

      s_anchor_int[tid] = static_cast<int>(bin_floor);
      s_anchor_frac[tid] = static_cast<float>(bin_anchor - bin_floor);
      s_ux_hi[tid] = static_cast<float>(dxa * inv_R_d);
      s_uy_hi[tid] = static_cast<float>(dya * inv_R_d);
      s_uz_hi[tid] = static_cast<float>(dza * inv_R_d);
      s_inv_R[tid] = static_cast<float>(inv_R_d);
    }
    __syncthreads();

    if (in_image) {
#pragma unroll 4
      for (int j = 0; j < chunk_size; ++j) {
        const float ux_hi = s_ux_hi[j];
        const float uy_hi = s_uy_hi[j];
        const float uz_hi = s_uz_hi[j];
        const float invR = s_inv_R[j];

        const float t1_hi = ux_hi * dx_hi;
        const float t2_hi = uy_hi * dy_hi;
        const float t3_hi = uz_hi * dz_hi;
        const float s12 = t1_hi + t2_hi;
        const float dot_hi = s12 + t3_hi;

        const float dsq = fmaf(dx_hi, dx_hi, fmaf(dy_hi, dy_hi, dz_hi * dz_hi));
        const float perp_sq = fmaf(-dot_hi, dot_hi, dsq);
        const float corr = perp_sq * 0.5f * invR * fmaf(-dot_hi, invR, 1.0f);
        const float dR_hi = dot_hi + corr;

        const float bin_loc = fmaf(dR_hi, dr_inv, s_anchor_frac[j]);
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
        const float vr = fmaf(w, v2r - v1r, v1r);
        const float vi = fmaf(w, v2i - v1i, v1i);

        const float base_r = phase_lut[2 * b1];
        const float base_i = phase_lut[2 * b1 + 1];
        float incr_sin = 0.0f;
        float incr_cos = 0.0f;
        __sincosf(pc_partial_dr * w, &incr_sin, &incr_cos);
        const float mfr = base_r * incr_cos - base_i * incr_sin;
        const float mfi = base_r * incr_sin + base_i * incr_cos;

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

// nvradar_block_anchored_expanded_throughput_phase_lut_vec: same math
// as the scalar throughput LUT kernel; vph/LUT as float2, shared
// packed as float4 + float2. No extra lattice-step args.
__global__ void __launch_bounds__(HOLOSCAN_RADAR_BP_LUT_PIXELS)
    backprojection_kernel_anchor_expanded_throughput_phase_lut_vec(  // NOLINT
        int fcount, int img_width, int img_height, const int Np, const int phase_center_stride,
        const double rmin, const double dr, const float dr_inv, const float pc_partial_dr,
        float* __restrict__ simg, const double* __restrict__ fvec,
        const double* __restrict__ ximg_grid, const double* __restrict__ yimg_grid,
        const double* __restrict__ zimg_grid, const double* __restrict__ phase_center,
        const double* __restrict__ R0, const float* __restrict__ vph,
        const float* __restrict__ phase_lut, int Nfft) {
  (void)fcount;
  (void)fvec;
  // Packed per-pulse shared state: one float4 {ux,uy,uz,1/R} + one
  // float2 {anchor_frac, __int_as_float(anchor_int)}.
  __shared__ float4 s_u[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float2 s_anchor[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ double s_ax_d;
  __shared__ double s_ay_d;
  __shared__ double s_az_d;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int xi = blockIdx.x * HOLOSCAN_RADAR_BP_LUT_SIDE + threadIdx.x;
  const int yi = blockIdx.y * HOLOSCAN_RADAR_BP_LUT_SIDE + threadIdx.y;
  const bool in_image = (xi < img_width) && (yi < img_height);
  const int image_loc = yi * img_width + xi;

  if (tid == 0) {
    const int ax = blockIdx.x * HOLOSCAN_RADAR_BP_LUT_SIDE + HOLOSCAN_RADAR_BP_LUT_SIDE / 2;
    const int ay = blockIdx.y * HOLOSCAN_RADAR_BP_LUT_SIDE + HOLOSCAN_RADAR_BP_LUT_SIDE / 2;
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
  const float dx_hi = in_image ? static_cast<float>(x_d - s_ax_d) : 0.0f;
  const float dy_hi = in_image ? static_cast<float>(y_d - s_ay_d) : 0.0f;
  const float dz_hi = in_image ? static_cast<float>(z_d - s_az_d) : 0.0f;

  const float2* __restrict__ vph2 = reinterpret_cast<const float2*>(vph);
  const float2* __restrict__ lut2 = reinterpret_cast<const float2*>(phase_lut);

  float acc_r = 0.0f;
  float acc_i = 0.0f;

  for (int chunk = 0; chunk < Np; chunk += HOLOSCAN_RADAR_BP_LUT_PIXELS) {
    const int remain = Np - chunk;
    const int chunk_size =
        remain < HOLOSCAN_RADAR_BP_LUT_PIXELS ? remain : HOLOSCAN_RADAR_BP_LUT_PIXELS;

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

      s_u[tid] = make_float4(static_cast<float>(dxa * inv_R_d), static_cast<float>(dya * inv_R_d),
                             static_cast<float>(dza * inv_R_d), static_cast<float>(inv_R_d));
      s_anchor[tid] = make_float2(static_cast<float>(bin_anchor - bin_floor),
                                  __int_as_float(static_cast<int>(bin_floor)));
    }
    __syncthreads();

    if (in_image) {
#pragma unroll 4
      for (int j = 0; j < chunk_size; ++j) {
        const float4 u = s_u[j];
        const float2 a = s_anchor[j];
        const float ux_hi = u.x;
        const float uy_hi = u.y;
        const float uz_hi = u.z;
        const float invR = u.w;

        // Keep the parent's exact summation order: rewriting this
        // as an explicit fmaf chain changes the fp32 rounding.
        const float t1_hi = ux_hi * dx_hi;
        const float t2_hi = uy_hi * dy_hi;
        const float t3_hi = uz_hi * dz_hi;
        const float s12 = t1_hi + t2_hi;
        const float dot_hi = s12 + t3_hi;

        const float dsq = fmaf(dx_hi, dx_hi, fmaf(dy_hi, dy_hi, dz_hi * dz_hi));
        const float perp_sq = fmaf(-dot_hi, dot_hi, dsq);
        const float corr = perp_sq * 0.5f * invR * fmaf(-dot_hi, invR, 1.0f);
        const float dR_hi = dot_hi + corr;

        const float bin_loc = fmaf(dR_hi, dr_inv, a.x);
        const float bf_floor = floorf(bin_loc);
        const int b1 = __float_as_int(a.y) + static_cast<int>(bf_floor);
        const float w = bin_loc - bf_floor;
        if ((b1 < 0) || (b1 >= Nfft - 1)) {
          continue;
        }

        const int p = chunk + j;
        const int row = p * Nfft;
        const float2 v1 = vph2[row + b1];
        const float2 v2 = vph2[row + b1 + 1];
        const float vr = fmaf(w, v2.x - v1.x, v1.x);
        const float vi = fmaf(w, v2.y - v1.y, v1.y);

        const float2 base = lut2[b1];
        float incr_sin = 0.0f;
        float incr_cos = 0.0f;
        __sincosf(pc_partial_dr * w, &incr_sin, &incr_cos);
        const float mfr = base.x * incr_cos - base.y * incr_sin;
        const float mfi = base.x * incr_sin + base.y * incr_cos;

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

// nvradar_block_anchored_expanded_throughput_phase_lut_vec_z0:
// planar-scene (image z == 0) specialization of _vec. zimg_grid is
// unused. Wrong on a non-planar grid. Same launch args as _vec.
__global__ void __launch_bounds__(HOLOSCAN_RADAR_BP_LUT_PIXELS)
    backprojection_kernel_anchor_expanded_throughput_phase_lut_vec_z0(  // NOLINT
        int fcount, int img_width, int img_height, const int Np, const int phase_center_stride,
        const double rmin, const double dr, const float dr_inv, const float pc_partial_dr,
        float* __restrict__ simg, const double* __restrict__ fvec,
        const double* __restrict__ ximg_grid, const double* __restrict__ yimg_grid,
        const double* __restrict__ zimg_grid, const double* __restrict__ phase_center,
        const double* __restrict__ R0, const float* __restrict__ vph,
        const float* __restrict__ phase_lut, int Nfft) {
  (void)fcount;
  (void)fvec;
  // zimg_grid is intentionally unused: this kernel assumes image z == 0.
  (void)zimg_grid;

  // Packed per-pulse shared state: float4 {ux, uy, 1/R, anchor_frac}
  // plus a separate int anchor bin.
  __shared__ float4 s_u[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ int s_anchor_int[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ double s_ax_d;
  __shared__ double s_ay_d;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int xi = blockIdx.x * HOLOSCAN_RADAR_BP_LUT_SIDE + threadIdx.x;
  const int yi = blockIdx.y * HOLOSCAN_RADAR_BP_LUT_SIDE + threadIdx.y;
  const bool in_image = (xi < img_width) && (yi < img_height);
  const int image_loc = yi * img_width + xi;

  if (tid == 0) {
    const int ax = blockIdx.x * HOLOSCAN_RADAR_BP_LUT_SIDE + HOLOSCAN_RADAR_BP_LUT_SIDE / 2;
    const int ay = blockIdx.y * HOLOSCAN_RADAR_BP_LUT_SIDE + HOLOSCAN_RADAR_BP_LUT_SIDE / 2;
    const int ax_c = ax < img_width ? ax : (img_width - 1);
    const int ay_c = ay < img_height ? ay : (img_height - 1);
    const int anchor_idx = ay_c * img_width + ax_c;
    s_ax_d = ximg_grid[anchor_idx];
    s_ay_d = yimg_grid[anchor_idx];
    // anchor z assumed 0 -- no zimg_grid read.
  }
  __syncthreads();

  double x_d = 0.0;
  double y_d = 0.0;
  if (in_image) {
    x_d = ximg_grid[image_loc];
    y_d = yimg_grid[image_loc];
  }
  const float dx_hi = in_image ? static_cast<float>(x_d - s_ax_d) : 0.0f;
  const float dy_hi = in_image ? static_cast<float>(y_d - s_ay_d) : 0.0f;

  const float2* __restrict__ vph2 = reinterpret_cast<const float2*>(vph);
  const float2* __restrict__ lut2 = reinterpret_cast<const float2*>(phase_lut);

  float acc_r = 0.0f;
  float acc_i = 0.0f;

  for (int chunk = 0; chunk < Np; chunk += HOLOSCAN_RADAR_BP_LUT_PIXELS) {
    const int remain = Np - chunk;
    const int chunk_size =
        remain < HOLOSCAN_RADAR_BP_LUT_PIXELS ? remain : HOLOSCAN_RADAR_BP_LUT_PIXELS;

    if (tid < chunk_size) {
      const int p = chunk + tid;
      const double pcx = phase_center[p];
      const double pcy = phase_center[p + phase_center_stride];
      const double pcz = phase_center[p + phase_center_stride * 2];
      const double R0_p_d = R0[p];

      const double dxa = s_ax_d - pcx;
      const double dya = s_ay_d - pcy;
      const double dza = -pcz;  // anchor z == 0
      const double R_anchor_pure = sqrt(dxa * dxa + dya * dya + dza * dza);
      const double R_anchor = R_anchor_pure - R0_p_d;
      const double bin_anchor = (R_anchor - rmin) / dr;
      const double bin_floor = floor(bin_anchor);
      const double inv_R_d = 1.0 / R_anchor_pure;

      // uz (dza * inv_R_d) is not stored: the inner loop never needs it
      // because dz == 0.
      s_u[tid] =
          make_float4(static_cast<float>(dxa * inv_R_d), static_cast<float>(dya * inv_R_d),
                      static_cast<float>(inv_R_d), static_cast<float>(bin_anchor - bin_floor));
      s_anchor_int[tid] = static_cast<int>(bin_floor);
    }
    __syncthreads();

    if (in_image) {
#pragma unroll 4
      for (int j = 0; j < chunk_size; ++j) {
        const float4 u = s_u[j];
        const float ux_hi = u.x;
        const float uy_hi = u.y;
        const float invR = u.z;
        const float frac = u.w;

        // dz == 0: the parent's t3 = uz*dz term and the dz*dz term in
        // dsq vanish. Summation order of the surviving terms matches
        // the parent exactly.
        const float t1_hi = ux_hi * dx_hi;
        const float t2_hi = uy_hi * dy_hi;
        const float dot_hi = t1_hi + t2_hi;

        const float dsq = fmaf(dx_hi, dx_hi, dy_hi * dy_hi);
        const float perp_sq = fmaf(-dot_hi, dot_hi, dsq);
        const float corr = perp_sq * 0.5f * invR * fmaf(-dot_hi, invR, 1.0f);
        const float dR_hi = dot_hi + corr;

        const float bin_loc = fmaf(dR_hi, dr_inv, frac);
        const float bf_floor = floorf(bin_loc);
        const int b1 = s_anchor_int[j] + static_cast<int>(bf_floor);
        const float w = bin_loc - bf_floor;
        if ((b1 < 0) || (b1 >= Nfft - 1)) {
          continue;
        }

        const int p = chunk + j;
        const int row = p * Nfft;
        const float2 v1 = vph2[row + b1];
        const float2 v2 = vph2[row + b1 + 1];
        const float vr = fmaf(w, v2.x - v1.x, v1.x);
        const float vi = fmaf(w, v2.y - v1.y, v1.y);

        const float2 base = lut2[b1];
        float incr_sin = 0.0f;
        float incr_cos = 0.0f;
        __sincosf(pc_partial_dr * w, &incr_sin, &incr_cos);
        const float mfr = base.x * incr_cos - base.y * incr_sin;
        const float mfi = base.x * incr_sin + base.y * incr_cos;

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

// nvradar_block_anchored_expanded_quality_phase_lut: TwoProduct dR +
// LUT (non-FMA interp / accumulate).
__global__ void backprojection_kernel_anchor_expanded_quality_phase_lut(
    int fcount, int img_width, int img_height, const int Np, const int phase_center_stride,
    const double rmin, const double dr, const float dr_inv, const float pc_partial_dr,
    float* __restrict__ simg, const double* __restrict__ fvec, const double* __restrict__ ximg_grid,
    const double* __restrict__ yimg_grid, const double* __restrict__ zimg_grid,
    const double* __restrict__ phase_center, const double* __restrict__ R0,
    const float* __restrict__ vph, const float* __restrict__ phase_lut, int Nfft) {
  (void)fcount;
  (void)fvec;
  // Phase comes from the LUT so we drop the parent's double-float dx/u
  // decomposition: the _lo shared arrays and the cross-term machinery
  // feeding dot_lo / dR_lo have no consumer in this variant.
  __shared__ int s_anchor_int[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_anchor_frac[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_ux_hi[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_uy_hi[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_uz_hi[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_inv_R[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ double s_ax_d;
  __shared__ double s_ay_d;
  __shared__ double s_az_d;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int xi = blockIdx.x * HOLOSCAN_RADAR_BP_LUT_SIDE + threadIdx.x;
  const int yi = blockIdx.y * HOLOSCAN_RADAR_BP_LUT_SIDE + threadIdx.y;
  const bool in_image = (xi < img_width) && (yi < img_height);
  const int image_loc = yi * img_width + xi;

  if (tid == 0) {
    const int ax = blockIdx.x * HOLOSCAN_RADAR_BP_LUT_SIDE + HOLOSCAN_RADAR_BP_LUT_SIDE / 2;
    const int ay = blockIdx.y * HOLOSCAN_RADAR_BP_LUT_SIDE + HOLOSCAN_RADAR_BP_LUT_SIDE / 2;
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
  const float dx_hi = in_image ? static_cast<float>(x_d - s_ax_d) : 0.0f;
  const float dy_hi = in_image ? static_cast<float>(y_d - s_ay_d) : 0.0f;
  const float dz_hi = in_image ? static_cast<float>(z_d - s_az_d) : 0.0f;

  float acc_r = 0.0f;
  float acc_i = 0.0f;

  for (int chunk = 0; chunk < Np; chunk += HOLOSCAN_RADAR_BP_LUT_PIXELS) {
    const int remain = Np - chunk;
    const int chunk_size =
        remain < HOLOSCAN_RADAR_BP_LUT_PIXELS ? remain : HOLOSCAN_RADAR_BP_LUT_PIXELS;

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

      s_anchor_int[tid] = static_cast<int>(bin_floor);
      s_anchor_frac[tid] = static_cast<float>(bin_anchor - bin_floor);
      s_ux_hi[tid] = static_cast<float>(dxa * inv_R_d);
      s_uy_hi[tid] = static_cast<float>(dya * inv_R_d);
      s_uz_hi[tid] = static_cast<float>(dza * inv_R_d);
      s_inv_R[tid] = static_cast<float>(inv_R_d);
    }
    __syncthreads();

    if (in_image) {
#pragma unroll 4
      for (int j = 0; j < chunk_size; ++j) {
        const float ux_hi = s_ux_hi[j];
        const float uy_hi = s_uy_hi[j];
        const float uz_hi = s_uz_hi[j];
        const float invR = s_inv_R[j];

        const float t1_hi = ux_hi * dx_hi;
        const float t2_hi = uy_hi * dy_hi;
        const float t3_hi = uz_hi * dz_hi;
        const float s12 = t1_hi + t2_hi;
        const float dot_hi = s12 + t3_hi;

        const float dsq = fmaf(dx_hi, dx_hi, fmaf(dy_hi, dy_hi, dz_hi * dz_hi));
        const float perp_sq = dsq - dot_hi * dot_hi;
        const float corr = perp_sq * 0.5f * invR * fmaf(-dot_hi, invR, 1.0f);
        const float dR_hi = dot_hi + corr;

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

        const float base_r = phase_lut[2 * b1];
        const float base_i = phase_lut[2 * b1 + 1];
        float incr_sin = 0.0f;
        float incr_cos = 0.0f;
        __sincosf(pc_partial_dr * w, &incr_sin, &incr_cos);
        const float mfr = base_r * incr_cos - base_i * incr_sin;
        const float mfi = base_r * incr_sin + base_i * incr_cos;

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

// nvradar_block_anchored_phase_lut_legacy: HQ block + LUT. fp32
// trajectory / mcp / R0 / grids. Extra mcp + mcp_stride.
__global__ void backprojection_kernel_hq_block_phase_lut(
    int fcount, int img_width, int img_height, const int Np, const int phase_center_stride,
    const int mcp_stride, const float rmin, const float dr_inv, const float pc_partial_dr,
    float* __restrict__ simg, const float* __restrict__ fvec, const float* __restrict__ ximg_grid,
    const float* __restrict__ yimg_grid, const float* __restrict__ zimg_grid,
    const float* __restrict__ phase_center, const float* __restrict__ mcp,
    const float* __restrict__ R0, const float* __restrict__ vph,
    const float* __restrict__ phase_lut, int Nfft) {
  (void)fcount;
  (void)fvec;
  __shared__ float s_ux[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_uy[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_uz[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_invR[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_R_rel[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_ax;
  __shared__ float s_ay;
  __shared__ float s_az;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int xi = blockIdx.x * HOLOSCAN_RADAR_BP_LUT_SIDE + threadIdx.x;
  const int yi = blockIdx.y * HOLOSCAN_RADAR_BP_LUT_SIDE + threadIdx.y;
  const bool in_image = (xi < img_width) && (yi < img_height);
  const int image_loc = yi * img_width + xi;

  if (tid == 0) {
    const int ax = blockIdx.x * HOLOSCAN_RADAR_BP_LUT_SIDE + HOLOSCAN_RADAR_BP_LUT_SIDE / 2;
    const int ay = blockIdx.y * HOLOSCAN_RADAR_BP_LUT_SIDE + HOLOSCAN_RADAR_BP_LUT_SIDE / 2;
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

  for (int chunk = 0; chunk < Np; chunk += HOLOSCAN_RADAR_BP_LUT_PIXELS) {
    const int remain = Np - chunk;
    const int chunk_size =
        remain < HOLOSCAN_RADAR_BP_LUT_PIXELS ? remain : HOLOSCAN_RADAR_BP_LUT_PIXELS;

    if (tid < chunk_size) {
      const int p = chunk + tid;
      const double ax_d = static_cast<double>(s_ax);
      const double ay_d = static_cast<double>(s_ay);
      const double az_d = static_cast<double>(s_az);
      const double pcx = static_cast<double>(phase_center[p]);
      const double pcy = static_cast<double>(phase_center[p + phase_center_stride]);
      const double pcz = static_cast<double>(phase_center[p + phase_center_stride * 2]);
      const double mx = static_cast<double>(mcp[p]);
      const double my = static_cast<double>(mcp[p + mcp_stride]);
      const double mz = static_cast<double>(mcp[p + mcp_stride * 2]);

      const double dx = ax_d - mx;
      const double dy = ay_d - my;
      const double dz = az_d - mz;
      const double vx = mx - pcx;
      const double vy = my - pcy;
      const double vz = mz - pcz;
      const double d_sq = dx * dx + dy * dy + dz * dz;
      const double v_dot_d = vx * dx + vy * dy + vz * dz;
      const double number = d_sq + 2.0 * v_dot_d;
      const double R0_p = static_cast<double>(R0[p]);
      const double R_raw = sqrt(number + R0_p * R0_p);
      const double R_rel = number / (R_raw + R0_p);

      const double dxp = ax_d - pcx;
      const double dyp = ay_d - pcy;
      const double dzp = az_d - pcz;
      const double R_anchor = sqrt(dxp * dxp + dyp * dyp + dzp * dzp);
      const double inv_R = 1.0 / R_anchor;

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
        const float dot = dx_a * ux + dy_a * uy + dz_a * uz;
        const float dsq = dx_a * dx_a + dy_a * dy_a + dz_a * dz_a;
        const float dR = dot + (dsq - dot * dot) * 0.5f * s_invR[j];

        const float R_rel_pixel = s_R_rel[j] + dR;
        const float bin = (R_rel_pixel - rmin) * dr_inv;
        if ((bin < 0.0f) || (bin >= static_cast<float>(Nfft - 1))) {
          continue;
        }
        const int b1 = static_cast<int>(floorf(bin));
        const float w = bin - floorf(bin);

        const int p = chunk + j;
        const int pulse_offset = p * Nfft * 2;
        const float v1r = vph[pulse_offset + b1 * 2];
        const float v1i = vph[pulse_offset + b1 * 2 + 1];
        const float v2r = vph[pulse_offset + b1 * 2 + 2];
        const float v2i = vph[pulse_offset + b1 * 2 + 3];
        const float vr = v1r + w * (v2r - v1r);
        const float vi = v1i + w * (v2i - v1i);

        const float base_r = phase_lut[2 * b1];
        const float base_i = phase_lut[2 * b1 + 1];
        float incr_sin = 0.0f;
        float incr_cos = 0.0f;
        __sincosf(pc_partial_dr * w, &incr_sin, &incr_cos);
        const float mfr = base_r * incr_cos - base_i * incr_sin;
        const float mfi = base_r * incr_sin + base_i * incr_cos;

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

// nvradar_block_anchored_fp64_trajectory_phase_lut: HQ block3 fp64
// trajectory + LUT. Extra mcp + mcp_stride; phase_center / mcp / R0
// are double*.
__global__ void backprojection_kernel_hq_block3_fp64in_phase_lut(
    int fcount, int img_width, int img_height, const int Np, const int phase_center_stride,
    const int mcp_stride, const float rmin, const float dr_inv, const float pc_partial_dr,
    float* __restrict__ simg, const float* __restrict__ fvec, const float* __restrict__ ximg_grid,
    const float* __restrict__ yimg_grid, const float* __restrict__ zimg_grid,
    const double* __restrict__ phase_center, const double* __restrict__ mcp,
    const double* __restrict__ R0, const float* __restrict__ vph,
    const float* __restrict__ phase_lut, int Nfft) {
  (void)fcount;
  (void)fvec;
  __shared__ float s_ux[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_uy[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_uz[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_invR[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_R_rel[HOLOSCAN_RADAR_BP_LUT_PIXELS];
  __shared__ float s_ax;
  __shared__ float s_ay;
  __shared__ float s_az;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int xi = blockIdx.x * HOLOSCAN_RADAR_BP_LUT_SIDE + threadIdx.x;
  const int yi = blockIdx.y * HOLOSCAN_RADAR_BP_LUT_SIDE + threadIdx.y;
  const bool in_image = (xi < img_width) && (yi < img_height);
  const int image_loc = yi * img_width + xi;

  if (tid == 0) {
    const int ax = blockIdx.x * HOLOSCAN_RADAR_BP_LUT_SIDE + HOLOSCAN_RADAR_BP_LUT_SIDE / 2;
    const int ay = blockIdx.y * HOLOSCAN_RADAR_BP_LUT_SIDE + HOLOSCAN_RADAR_BP_LUT_SIDE / 2;
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

  for (int chunk = 0; chunk < Np; chunk += HOLOSCAN_RADAR_BP_LUT_PIXELS) {
    const int remain = Np - chunk;
    const int chunk_size =
        remain < HOLOSCAN_RADAR_BP_LUT_PIXELS ? remain : HOLOSCAN_RADAR_BP_LUT_PIXELS;

    if (tid < chunk_size) {
      const int p = chunk + tid;
      const double ax_d = static_cast<double>(s_ax);
      const double ay_d = static_cast<double>(s_ay);
      const double az_d = static_cast<double>(s_az);
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
      const double number = d_sq + 2.0 * v_dot_d;
      const double R0_p = R0[p];
      const double R_raw = sqrt(number + R0_p * R0_p);
      const double R_rel = number / (R_raw + R0_p);

      const double dxp = ax_d - pcx;
      const double dyp = ay_d - pcy;
      const double dzp = az_d - pcz;
      const double R_anchor = sqrt(dxp * dxp + dyp * dyp + dzp * dzp);
      const double inv_R = 1.0 / R_anchor;

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
        const float dR = dot + perp_sq * 0.5f * inv_R * (1.0f - dot * inv_R);

        const float R_rel_pixel = s_R_rel[j] + dR;
        const float bin = (R_rel_pixel - rmin) * dr_inv;
        if ((bin < 0.0f) || (bin >= static_cast<float>(Nfft - 1))) {
          continue;
        }
        const int b1 = static_cast<int>(floorf(bin));
        const float w = bin - floorf(bin);

        const int p = chunk + j;
        const int pulse_offset = p * Nfft * 2;
        const float v1r = vph[pulse_offset + b1 * 2];
        const float v1i = vph[pulse_offset + b1 * 2 + 1];
        const float v2r = vph[pulse_offset + b1 * 2 + 2];
        const float v2i = vph[pulse_offset + b1 * 2 + 3];
        const float vr = v1r + w * (v2r - v1r);
        const float vi = v1i + w * (v2i - v1i);

        const float base_r = phase_lut[2 * b1];
        const float base_i = phase_lut[2 * b1 + 1];
        float incr_sin = 0.0f;
        float incr_cos = 0.0f;
        __sincosf(pc_partial_dr * w, &incr_sin, &incr_cos);
        const float mfr = base_r * incr_cos - base_i * incr_sin;
        const float mfi = base_r * incr_sin + base_i * incr_cos;

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

#undef HOLOSCAN_RADAR_BP_LUT_SIDE
#undef HOLOSCAN_RADAR_BP_LUT_PIXELS

}  // namespace holoscan::radar::detail

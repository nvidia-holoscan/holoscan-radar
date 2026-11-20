// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Copied from NVRadar backprojection CUDA kernels
// (modules/edge/python/nvradar/edge/operators/backprojection_image_formation/bp_kernels.py)
// at SHA 37c1cb29355bcfe256ae728f51dd47071847d949. OSRB relicensing of the
// copies is recorded in this operator's README.md. Naive / HQ family
// only. Do not copy backprojection_kernel (legacy float bin/w).

#pragma once

namespace holoscan::radar::detail {

__global__ void backprojection_kernel_fp32(
    int fcount, int img_width, int img_height, const int Np, const int phase_center_stride,
    const float rmin, const float dr_inv, const float pc_partial, float* __restrict__ simg,
    const float* __restrict__ fvec, const float* __restrict__ ximg_grid,
    const float* __restrict__ yimg_grid, const float* __restrict__ zimg_grid,
    const float* __restrict__ phase_center, const float* __restrict__ R0,
    const float* __restrict__ vph, int Nfft) {
  (void)fcount;
  (void)fvec;
  const int xi = blockIdx.x * blockDim.x + threadIdx.x;
  const int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if ((xi >= img_width) || (yi >= img_height)) {
    return;
  }
  const int image_loc = yi * img_width + xi;
  const float x = ximg_grid[image_loc];
  const float y = yimg_grid[image_loc];
  const float z = zimg_grid[image_loc];
  float acc_r = 0.0f;
  float acc_i = 0.0f;
#pragma unroll 4
  for (int p = 0; p < Np; ++p) {
    const int pulse_offset = p * Nfft * 2;
    const float px = phase_center[p];
    const float py = phase_center[p + phase_center_stride];
    const float pz = phase_center[p + phase_center_stride * 2];
    const float pR0 = R0[p];
    float R = sqrtf((x - px) * (x - px) + (y - py) * (y - py) + (z - pz) * (z - pz));
    R = R - pR0;
    const float bin = (R - rmin) * dr_inv;
    if ((bin < 0) || (bin >= Nfft - 1)) {
      continue;
    }
    const int b1 = static_cast<int>(floorf(bin));
    const int b2 = b1 + 1;
    const float w = bin - floorf(bin);
    float mfr = 0.0f;
    float mfi = 0.0f;
    sincosf(pc_partial * R, &mfi, &mfr);
    const float v1r = vph[pulse_offset + b1 * 2];
    const float v1i = vph[pulse_offset + b1 * 2 + 1];
    const float v2r = vph[pulse_offset + b2 * 2];
    const float v2i = vph[pulse_offset + b2 * 2 + 1];
    const float vr = v1r + w * (v2r - v1r);
    const float vi = v1i + w * (v2i - v1i);
    acc_r += vr * mfr - vi * mfi;
    acc_i += vr * mfi + vi * mfr;
  }
  simg[2 * image_loc] += acc_r;
  simg[2 * image_loc + 1] += acc_i;
}

// HQ: range via a^2 - b^2 identity so R_raw - R0 does not cancel.
__global__ void backprojection_kernel_fp32_hq(
    int fcount, int img_width, int img_height, const int Np, const int phase_center_stride,
    const int mcp_stride, const float rmin, const float dr_inv, const float pc_partial,
    float* __restrict__ simg, const float* __restrict__ fvec, const float* __restrict__ ximg_grid,
    const float* __restrict__ yimg_grid, const float* __restrict__ zimg_grid,
    const float* __restrict__ phase_center, const float* __restrict__ mcp,
    const float* __restrict__ R0, const float* __restrict__ vph, int Nfft) {
  (void)fcount;
  (void)fvec;
  const int xi = blockIdx.x * blockDim.x + threadIdx.x;
  const int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if ((xi >= img_width) || (yi >= img_height)) {
    return;
  }
  const int image_loc = yi * img_width + xi;
  const float x = ximg_grid[image_loc];
  const float y = yimg_grid[image_loc];
  const float z = zimg_grid[image_loc];
  float acc_r = 0.0f;
  float acc_i = 0.0f;
#pragma unroll 4
  for (int p = 0; p < Np; ++p) {
    const int pulse_offset = p * Nfft * 2;
    const float mx = mcp[p];
    const float my = mcp[p + mcp_stride];
    const float mz = mcp[p + mcp_stride * 2];
    const float dx = x - mx;
    const float dy = y - my;
    const float dz = z - mz;
    const float d_sq = dx * dx + dy * dy + dz * dz;
    const float number = d_sq + 2.0f * (dx * (mx - phase_center[p]) +
                                        dy * (my - phase_center[p + phase_center_stride]) +
                                        dz * (mz - phase_center[p + phase_center_stride * 2]));
    const float R0_val = R0[p];
    const float R_raw = sqrtf(number + R0_val * R0_val);
    const float R = number / (R_raw + R0_val);
    const float bin = (R - rmin) * dr_inv;
    if ((bin < 0) || (bin >= Nfft - 1)) {
      continue;
    }
    const int b1 = static_cast<int>(floorf(bin));
    const int b2 = b1 + 1;
    const float w = bin - floorf(bin);
    float mfr = 0.0f;
    float mfi = 0.0f;
    sincosf(pc_partial * R, &mfi, &mfr);
    const float v1r = vph[pulse_offset + b1 * 2];
    const float v1i = vph[pulse_offset + b1 * 2 + 1];
    const float v2r = vph[pulse_offset + b2 * 2];
    const float v2i = vph[pulse_offset + b2 * 2 + 1];
    const float vr = v1r + w * (v2r - v1r);
    const float vi = v1i + w * (v2i - v1i);
    acc_r += vr * mfr - vi * mfi;
    acc_i += vr * mfi + vi * mfr;
  }
  simg[2 * image_loc] += acc_r;
  simg[2 * image_loc + 1] += acc_i;
}

// HQ-fast: rsqrtf + __fdividef. Same identity as _hq.
__global__ void backprojection_kernel_fp32_hq_fast(
    int fcount, int img_width, int img_height, const int Np, const int phase_center_stride,
    const int mcp_stride, const float rmin, const float dr_inv, const float pc_partial,
    float* __restrict__ simg, const float* __restrict__ fvec, const float* __restrict__ ximg_grid,
    const float* __restrict__ yimg_grid, const float* __restrict__ zimg_grid,
    const float* __restrict__ phase_center, const float* __restrict__ mcp,
    const float* __restrict__ R0, const float* __restrict__ vph, int Nfft) {
  (void)fcount;
  (void)fvec;
  const int xi = blockIdx.x * blockDim.x + threadIdx.x;
  const int yi = blockIdx.y * blockDim.y + threadIdx.y;
  if ((xi >= img_width) || (yi >= img_height)) {
    return;
  }
  const int image_loc = yi * img_width + xi;
  const float x = ximg_grid[image_loc];
  const float y = yimg_grid[image_loc];
  const float z = zimg_grid[image_loc];
  float acc_r = 0.0f;
  float acc_i = 0.0f;
#pragma unroll 4
  for (int p = 0; p < Np; ++p) {
    const int pulse_offset = p * Nfft * 2;
    const float mx = mcp[p];
    const float my = mcp[p + mcp_stride];
    const float mz = mcp[p + mcp_stride * 2];
    const float dx = x - mx;
    const float dy = y - my;
    const float dz = z - mz;
    const float d_sq = dx * dx + dy * dy + dz * dz;
    const float number = d_sq + 2.0f * (dx * (mx - phase_center[p]) +
                                        dy * (my - phase_center[p + phase_center_stride]) +
                                        dz * (mz - phase_center[p + phase_center_stride * 2]));
    const float R0_val = R0[p];
    const float sum = number + R0_val * R0_val;
    const float R_raw = sum * rsqrtf(sum);
    const float R = __fdividef(number, R_raw + R0_val);
    const float bin = (R - rmin) * dr_inv;
    if ((bin < 0) || (bin >= Nfft - 1)) {
      continue;
    }
    const int b1 = static_cast<int>(floorf(bin));
    const int b2 = b1 + 1;
    const float w = bin - floorf(bin);
    float mfr = 0.0f;
    float mfi = 0.0f;
    sincosf(pc_partial * R, &mfi, &mfr);
    const float v1r = vph[pulse_offset + b1 * 2];
    const float v1i = vph[pulse_offset + b1 * 2 + 1];
    const float v2r = vph[pulse_offset + b2 * 2];
    const float v2i = vph[pulse_offset + b2 * 2 + 1];
    const float vr = v1r + w * (v2r - v1r);
    const float vi = v1i + w * (v2i - v1i);
    acc_r += vr * mfr - vi * mfi;
    acc_i += vr * mfi + vi * mfr;
  }
  simg[2 * image_loc] += acc_r;
  simg[2 * image_loc + 1] += acc_i;
}

// nvradar_block_anchored: HQ identity at a per-block fp64 anchor, Taylor
// expansion in fp32 for the pixel delta.
#define HOLOSCAN_RADAR_BP_HQ_SIDE 16
#define HOLOSCAN_RADAR_BP_HQ_PIXELS (HOLOSCAN_RADAR_BP_HQ_SIDE * HOLOSCAN_RADAR_BP_HQ_SIDE)

__global__ void backprojection_kernel_fp32_hq_block(
    int fcount, int img_width, int img_height, const int Np, const int phase_center_stride,
    const int mcp_stride, const float rmin, const float dr_inv, const float pc_partial,
    float* __restrict__ simg, const float* __restrict__ fvec, const float* __restrict__ ximg_grid,
    const float* __restrict__ yimg_grid, const float* __restrict__ zimg_grid,
    const float* __restrict__ phase_center, const float* __restrict__ mcp,
    const float* __restrict__ R0, const float* __restrict__ vph, int Nfft) {
  (void)fcount;
  (void)fvec;
  const float INV_TWOPI_F = 0.15915494309189534f;
  const float TWOPI_F = 6.2831853071795865f;
  const double INV_TWOPI_D = 0.15915494309189534560822210096757;
  const double TWOPI_D = 6.283185307179586476925286766559;

  __shared__ float s_phi[HOLOSCAN_RADAR_BP_HQ_PIXELS];
  __shared__ float s_ux[HOLOSCAN_RADAR_BP_HQ_PIXELS];
  __shared__ float s_uy[HOLOSCAN_RADAR_BP_HQ_PIXELS];
  __shared__ float s_uz[HOLOSCAN_RADAR_BP_HQ_PIXELS];
  __shared__ float s_invR[HOLOSCAN_RADAR_BP_HQ_PIXELS];
  __shared__ float s_R_rel[HOLOSCAN_RADAR_BP_HQ_PIXELS];
  __shared__ float s_ax;
  __shared__ float s_ay;
  __shared__ float s_az;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int xi = blockIdx.x * HOLOSCAN_RADAR_BP_HQ_SIDE + threadIdx.x;
  const int yi = blockIdx.y * HOLOSCAN_RADAR_BP_HQ_SIDE + threadIdx.y;
  const bool in_image = (xi < img_width) && (yi < img_height);
  const int image_loc = yi * img_width + xi;

  if (tid == 0) {
    const int ax = blockIdx.x * HOLOSCAN_RADAR_BP_HQ_SIDE + HOLOSCAN_RADAR_BP_HQ_SIDE / 2;
    const int ay = blockIdx.y * HOLOSCAN_RADAR_BP_HQ_SIDE + HOLOSCAN_RADAR_BP_HQ_SIDE / 2;
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

  for (int chunk = 0; chunk < Np; chunk += HOLOSCAN_RADAR_BP_HQ_PIXELS) {
    const int remain = Np - chunk;
    const int chunk_size =
        remain < HOLOSCAN_RADAR_BP_HQ_PIXELS ? remain : HOLOSCAN_RADAR_BP_HQ_PIXELS;
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

#undef HOLOSCAN_RADAR_BP_HQ_SIDE
#undef HOLOSCAN_RADAR_BP_HQ_PIXELS

}  // namespace holoscan::radar::detail

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include "../../main/c/simd/vector_simd.h"

#define M 1
#define N 6656
#define K 2048
#define ITERATIONS 20

static short bf16_value(int index) {
    union {
        float f;
        unsigned int bits;
    } value;
    value.f = (float) (((index * 37) % 257) - 128) / 32.0f;
    return (short) (value.bits >> 16);
}

static float f32_value(int index) {
    return (float) (((index * 29) % 193) - 96) / 24.0f;
}

static double elapsed_seconds(const struct timespec *start, const struct timespec *end) {
    return (double) (end->tv_sec - start->tv_sec)
            + (double) (end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

int main(void) {
    static float a[M * K];
    static short b[N * K];
    static float existing[M * N];
    static float transa[M * N];

    for (int i = 0; i < M * K; i++) a[i] = f32_value(i);
    for (int i = 0; i < N * K; i++) b[i] = bf16_value(10000 + i);

    gemm_f32_bf16(0, a, 0, b, 0, NULL, existing, 0, M, 0, N, K, K, K, N);
    gemm_f32_bf16_transa(a, 0, b, 0, NULL, transa, 0, M, 0, N, K, K, K, N);

    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        max_error = fmaxf(max_error, fabsf(existing[i] - transa[i]));
    }
    printf("max_error=%g\n", max_error);
    assert(max_error < 0.01f);

    volatile float checksum = 0.0f;
    struct timespec start;
    struct timespec end;

    for (int i = 0; i < 10; i++) {
        gemm_f32_bf16(0, a, 0, b, 0, NULL, existing, 0, M, 0, N, K, K, K, N);
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) {
        gemm_f32_bf16(0, a, 0, b, 0, NULL, existing, 0, M, 0, N, K, K, K, N);
        checksum += existing[i % N];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double existing_seconds = elapsed_seconds(&start, &end);

    for (int i = 0; i < 10; i++) {
        gemm_f32_bf16_transa(a, 0, b, 0, NULL, transa, 0, M, 0, N, K, K, K, N);
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) {
        gemm_f32_bf16_transa(a, 0, b, 0, NULL, transa, 0, M, 0, N, K, K, K, N);
        checksum += transa[i % N];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double transa_seconds = elapsed_seconds(&start, &end);

    printf("existing_f32_bf16_gemm_ms=%f\n", existing_seconds * 1000.0 / ITERATIONS);
    printf("transa_f32_bf16_gemm_ms=%f\n", transa_seconds * 1000.0 / ITERATIONS);
    printf("checksum=%f\n", checksum);
    return 0;
}

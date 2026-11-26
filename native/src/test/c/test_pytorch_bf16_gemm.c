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

static double elapsed_seconds(const struct timespec *start, const struct timespec *end) {
    return (double) (end->tv_sec - start->tv_sec)
            + (double) (end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

int main(void) {
    static short a[M * K];
    static short b[N * K];
    static float existing[M * N];
    static float pytorch[M * N];
    static float pytorch_transa[M * N];

    for (int i = 0; i < M * K; i++) a[i] = bf16_value(i);
    for (int i = 0; i < N * K; i++) b[i] = bf16_value(10000 + i);

    gemm_bf16(0, a, 0, b, 0, NULL, existing, 0, M, 0, N, K, K, K, N);
    gemm_bf16_pytorch(a, 0, b, 0, pytorch, 0, M, 0, N, K, K, K, N);
    gemm_bf16_pytorch_transa(a, 0, b, 0, pytorch_transa, 0, M, 0, N, K, K, K, N);

    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        max_error = fmaxf(max_error, fabsf(existing[i] - pytorch[i]));
    }
    printf("max_error=%g\n", max_error);
    assert(max_error < 0.01f);

    float transa_max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        transa_max_error = fmaxf(transa_max_error, fabsf(existing[i] - pytorch_transa[i]));
    }
    printf("transa_max_error=%g\n", transa_max_error);
    assert(transa_max_error < 0.01f);

    volatile float checksum = 0.0f;
    struct timespec start;
    struct timespec end;

    for (int i = 0; i < 10; i++) {
        gemm_bf16(0, a, 0, b, 0, NULL, existing, 0, M, 0, N, K, K, K, N);
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) {
        gemm_bf16(0, a, 0, b, 0, NULL, existing, 0, M, 0, N, K, K, K, N);
        checksum += existing[i % N];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double existing_seconds = elapsed_seconds(&start, &end);

    for (int i = 0; i < 10; i++) {
        gemm_bf16_pytorch(a, 0, b, 0, pytorch, 0, M, 0, N, K, K, K, N);
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) {
        gemm_bf16_pytorch(a, 0, b, 0, pytorch, 0, M, 0, N, K, K, K, N);
        checksum += pytorch[i % N];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double pytorch_seconds = elapsed_seconds(&start, &end);

    for (int i = 0; i < 10; i++) {
        gemm_bf16_pytorch_transa(a, 0, b, 0, pytorch_transa, 0, M, 0, N, K, K, K, N);
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) {
        gemm_bf16_pytorch_transa(a, 0, b, 0, pytorch_transa, 0, M, 0, N, K, K, K, N);
        checksum += pytorch_transa[i % N];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double pytorch_transa_seconds = elapsed_seconds(&start, &end);

    printf("existing_bf16_gemm_ms=%f\n", existing_seconds * 1000.0 / ITERATIONS);
    printf("pytorch_bf16_gemm_ms=%f\n", pytorch_seconds * 1000.0 / ITERATIONS);
    printf("pytorch_transa_bf16_gemm_ms=%f\n", pytorch_transa_seconds * 1000.0 / ITERATIONS);
    printf("checksum=%f\n", checksum);
    return 0;
}

#include <Accelerate/Accelerate.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#include "../../main/c/simd/vector_simd.h"

#define M 1
#define N 64
#define K 128
#define ITERATIONS 100

static float value_at(int index) {
    return (float) (((index * 37) % 257) - 128) / 32.0f;
}

static double elapsed_seconds(const struct timespec *start, const struct timespec *end) {
    return (double) (end->tv_sec - start->tv_sec)
            + (double) (end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

int main(void) {
    // A is row-major [M, K], B is row-major [N, K], matching gemm_f32's
    // output-row weight layout.
    static float a[M * K];
    static float b[K * N];
    static float native_result[M * N];
    static float accelerate_result[M * N];
    static float accelerate_gemv_result[N];

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < K; col++) {
            a[row * K + col] = value_at(row * K + col);
        }
    }
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < K; col++) {
            b[row * K + col] = value_at(10000 + row * K + col);
        }
    }

    gemm_f32(0, a, 0, b, 0, native_result, 0, M, 0, N, K, K, K, N);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f,
                a, K,
                b, K,
                0.0f,
                accelerate_result, N);

    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        max_error = fmaxf(max_error, fabsf(native_result[i] - accelerate_result[i]));
    }
    printf("max_error=%g\n", max_error);
    assert(max_error < 0.01f);

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                N, K,
                1.0f,
                b, K,
                a, 1,
                0.0f,
                accelerate_gemv_result, 1);
    float gemv_max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        gemv_max_error = fmaxf(gemv_max_error,
                               fabsf(native_result[i] - accelerate_gemv_result[i]));
    }
    printf("gemv_max_error=%g\n", gemv_max_error);
    assert(gemv_max_error < 0.01f);

    volatile float checksum = 0.0f;
    struct timespec start;
    struct timespec end;

    for (int i = 0; i < 10; i++) {
        gemm_f32(0, a, 0, b, 0, native_result, 0, M, 0, N, K, K, K, N);
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) {
        gemm_f32(0, a, 0, b, 0, native_result, 0, M, 0, N, K, K, K, N);
        checksum += native_result[i % (M * N)];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double native_seconds = elapsed_seconds(&start, &end);

    for (int i = 0; i < 10; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N, K, 1.0f, a, K, b, K, 0.0f, accelerate_result, N);
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N, K, 1.0f, a, K, b, K, 0.0f, accelerate_result, N);
        checksum += accelerate_result[i % (M * N)];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double accelerate_seconds = elapsed_seconds(&start, &end);

    for (int i = 0; i < 10; i++) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    N, K, 1.0f, b, K, a, 1, 0.0f, accelerate_gemv_result, 1);
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    N, K, 1.0f, b, K, a, 1, 0.0f, accelerate_gemv_result, 1);
        checksum += accelerate_gemv_result[i % N];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double accelerate_gemv_seconds = elapsed_seconds(&start, &end);

    printf("native_gemm_ms=%f\n", native_seconds * 1000.0 / ITERATIONS);
    printf("accelerate_sgemm_ms=%f\n", accelerate_seconds * 1000.0 / ITERATIONS);
    printf("accelerate_sgemv_ms=%f\n", accelerate_gemv_seconds * 1000.0 / ITERATIONS);
    printf("checksum=%f\n", checksum);
    return 0;
}

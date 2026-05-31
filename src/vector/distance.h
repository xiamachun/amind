#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define CLAWMIND_HAS_NEON 1
#else
#define CLAWMIND_HAS_NEON 0
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#define CLAWMIND_HAS_AVX2 1
#else
#define CLAWMIND_HAS_AVX2 0
#endif

#if defined(__SSE__) && !CLAWMIND_HAS_AVX2 && !CLAWMIND_HAS_NEON
#include <xmmintrin.h>
#include <pmmintrin.h>
#define CLAWMIND_HAS_SSE 1
#else
#define CLAWMIND_HAS_SSE 0
#endif

namespace amind {

/// Distance metric used by the HNSW index.
enum class DistanceMetric : uint8_t {
    Cosine = 0,
    L2 = 1,
    InnerProduct = 2,
};

/// Compute the cosine similarity between two vectors.
/// Returns a value in [-1, 1], where 1 means identical direction.
/// Throws if dimensions mismatch or either vector has zero magnitude.
inline float cosineSimilarity(std::span<const float> vectorA,
                              std::span<const float> vectorB) {
    if (vectorA.size() != vectorB.size()) {
        throw std::invalid_argument("cosineSimilarity: dimension mismatch");
    }

    const size_t dimension = vectorA.size();
    if (dimension == 0) {
        throw std::invalid_argument("cosineSimilarity: empty vectors");
    }

    float dotProduct = 0.0f;
    float normSquaredA = 0.0f;
    float normSquaredB = 0.0f;

    size_t index = 0;

#if CLAWMIND_HAS_AVX2
    // AVX2 SIMD path: process 8 floats at a time
    __m256 sumDot256 = _mm256_setzero_ps();
    __m256 sumNormA256 = _mm256_setzero_ps();
    __m256 sumNormB256 = _mm256_setzero_ps();

    for (; index + 8 <= dimension; index += 8) {
        __m256 chunkA = _mm256_loadu_ps(&vectorA[index]);
        __m256 chunkB = _mm256_loadu_ps(&vectorB[index]);

        sumDot256 = _mm256_fmadd_ps(chunkA, chunkB, sumDot256);
        sumNormA256 = _mm256_fmadd_ps(chunkA, chunkA, sumNormA256);
        sumNormB256 = _mm256_fmadd_ps(chunkB, chunkB, sumNormB256);
    }

    // Horizontal sum of 8 floats
    __m128 hiDot = _mm256_extractf128_ps(sumDot256, 1);
    __m128 loDot = _mm256_castps256_ps128(sumDot256);
    __m128 sumDot128 = _mm_add_ps(hiDot, loDot);
    sumDot128 = _mm_hadd_ps(sumDot128, sumDot128);
    sumDot128 = _mm_hadd_ps(sumDot128, sumDot128);
    dotProduct = _mm_cvtss_f32(sumDot128);

    __m128 hiA = _mm256_extractf128_ps(sumNormA256, 1);
    __m128 loA = _mm256_castps256_ps128(sumNormA256);
    __m128 sumA128 = _mm_add_ps(hiA, loA);
    sumA128 = _mm_hadd_ps(sumA128, sumA128);
    sumA128 = _mm_hadd_ps(sumA128, sumA128);
    normSquaredA = _mm_cvtss_f32(sumA128);

    __m128 hiB = _mm256_extractf128_ps(sumNormB256, 1);
    __m128 loB = _mm256_castps256_ps128(sumNormB256);
    __m128 sumB128 = _mm_add_ps(hiB, loB);
    sumB128 = _mm_hadd_ps(sumB128, sumB128);
    sumB128 = _mm_hadd_ps(sumB128, sumB128);
    normSquaredB = _mm_cvtss_f32(sumB128);

#elif CLAWMIND_HAS_SSE
    // SSE SIMD path: process 4 floats at a time
    __m128 sumDotSSE = _mm_setzero_ps();
    __m128 sumNormASSE = _mm_setzero_ps();
    __m128 sumNormBSSE = _mm_setzero_ps();

    for (; index + 4 <= dimension; index += 4) {
        __m128 chunkA = _mm_loadu_ps(&vectorA[index]);
        __m128 chunkB = _mm_loadu_ps(&vectorB[index]);

        sumDotSSE = _mm_add_ps(sumDotSSE, _mm_mul_ps(chunkA, chunkB));
        sumNormASSE = _mm_add_ps(sumNormASSE, _mm_mul_ps(chunkA, chunkA));
        sumNormBSSE = _mm_add_ps(sumNormBSSE, _mm_mul_ps(chunkB, chunkB));
    }

    sumDotSSE = _mm_hadd_ps(sumDotSSE, sumDotSSE);
    sumDotSSE = _mm_hadd_ps(sumDotSSE, sumDotSSE);
    dotProduct = _mm_cvtss_f32(sumDotSSE);

    sumNormASSE = _mm_hadd_ps(sumNormASSE, sumNormASSE);
    sumNormASSE = _mm_hadd_ps(sumNormASSE, sumNormASSE);
    normSquaredA = _mm_cvtss_f32(sumNormASSE);

    sumNormBSSE = _mm_hadd_ps(sumNormBSSE, sumNormBSSE);
    sumNormBSSE = _mm_hadd_ps(sumNormBSSE, sumNormBSSE);
    normSquaredB = _mm_cvtss_f32(sumNormBSSE);

#elif CLAWMIND_HAS_NEON
    // NEON SIMD path: process 4 floats at a time
    float32x4_t sumDot = vdupq_n_f32(0.0f);
    float32x4_t sumNormA = vdupq_n_f32(0.0f);
    float32x4_t sumNormB = vdupq_n_f32(0.0f);

    for (; index + 4 <= dimension; index += 4) {
        float32x4_t chunkA = vld1q_f32(&vectorA[index]);
        float32x4_t chunkB = vld1q_f32(&vectorB[index]);

        sumDot = vfmaq_f32(sumDot, chunkA, chunkB);
        sumNormA = vfmaq_f32(sumNormA, chunkA, chunkA);
        sumNormB = vfmaq_f32(sumNormB, chunkB, chunkB);
    }

    dotProduct = vaddvq_f32(sumDot);
    normSquaredA = vaddvq_f32(sumNormA);
    normSquaredB = vaddvq_f32(sumNormB);
#endif

    // Scalar tail (handles remaining elements after SIMD, or full scalar path)
    for (; index < dimension; ++index) {
        dotProduct += vectorA[index] * vectorB[index];
        normSquaredA += vectorA[index] * vectorA[index];
        normSquaredB += vectorB[index] * vectorB[index];
    }

    float denominator = std::sqrt(normSquaredA) * std::sqrt(normSquaredB);
    if (denominator < 1e-10f) {
        return 0.0f;  // degenerate case: zero-magnitude vector
    }

    return dotProduct / denominator;
}

/// Cosine distance = 1 - cosine_similarity.
/// Returns a value in [0, 2], where 0 means identical direction.
inline float cosineDistance(std::span<const float> vectorA,
                            std::span<const float> vectorB) {
    return 1.0f - cosineSimilarity(vectorA, vectorB);
}

/// Squared L2 (Euclidean) distance between two vectors.
/// Avoids the sqrt for performance in comparisons.
inline float l2DistanceSquared(std::span<const float> vectorA,
                                std::span<const float> vectorB) {
    if (vectorA.size() != vectorB.size()) {
        throw std::invalid_argument("l2DistanceSquared: dimension mismatch");
    }

    const size_t dimension = vectorA.size();
    float sumSquared = 0.0f;
    size_t index = 0;

#if CLAWMIND_HAS_AVX2
    __m256 sumVec256 = _mm256_setzero_ps();

    for (; index + 8 <= dimension; index += 8) {
        __m256 chunkA = _mm256_loadu_ps(&vectorA[index]);
        __m256 chunkB = _mm256_loadu_ps(&vectorB[index]);
        __m256 diff = _mm256_sub_ps(chunkA, chunkB);
        sumVec256 = _mm256_fmadd_ps(diff, diff, sumVec256);
    }

    __m128 hi = _mm256_extractf128_ps(sumVec256, 1);
    __m128 lo = _mm256_castps256_ps128(sumVec256);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sumSquared = _mm_cvtss_f32(sum128);

#elif CLAWMIND_HAS_SSE
    __m128 sumVecSSE = _mm_setzero_ps();

    for (; index + 4 <= dimension; index += 4) {
        __m128 chunkA = _mm_loadu_ps(&vectorA[index]);
        __m128 chunkB = _mm_loadu_ps(&vectorB[index]);
        __m128 diff = _mm_sub_ps(chunkA, chunkB);
        sumVecSSE = _mm_add_ps(sumVecSSE, _mm_mul_ps(diff, diff));
    }

    sumVecSSE = _mm_hadd_ps(sumVecSSE, sumVecSSE);
    sumVecSSE = _mm_hadd_ps(sumVecSSE, sumVecSSE);
    sumSquared = _mm_cvtss_f32(sumVecSSE);

#elif CLAWMIND_HAS_NEON
    float32x4_t sumVec = vdupq_n_f32(0.0f);

    for (; index + 4 <= dimension; index += 4) {
        float32x4_t chunkA = vld1q_f32(&vectorA[index]);
        float32x4_t chunkB = vld1q_f32(&vectorB[index]);
        float32x4_t diff = vsubq_f32(chunkA, chunkB);
        sumVec = vfmaq_f32(sumVec, diff, diff);
    }

    sumSquared = vaddvq_f32(sumVec);
#endif

    for (; index < dimension; ++index) {
        float diff = vectorA[index] - vectorB[index];
        sumSquared += diff * diff;
    }

    return sumSquared;
}

/// L2 (Euclidean) distance between two vectors.
inline float l2Distance(std::span<const float> vectorA,
                         std::span<const float> vectorB) {
    return std::sqrt(l2DistanceSquared(vectorA, vectorB));
}

/// Inner product (dot product) between two vectors.
inline float innerProduct(std::span<const float> vectorA,
                           std::span<const float> vectorB) {
    if (vectorA.size() != vectorB.size()) {
        throw std::invalid_argument("innerProduct: dimension mismatch");
    }

    const size_t dimension = vectorA.size();
    float dotProduct = 0.0f;
    size_t index = 0;

#if CLAWMIND_HAS_AVX2
    __m256 sumVec256 = _mm256_setzero_ps();

    for (; index + 8 <= dimension; index += 8) {
        __m256 chunkA = _mm256_loadu_ps(&vectorA[index]);
        __m256 chunkB = _mm256_loadu_ps(&vectorB[index]);
        sumVec256 = _mm256_fmadd_ps(chunkA, chunkB, sumVec256);
    }

    __m128 hi = _mm256_extractf128_ps(sumVec256, 1);
    __m128 lo = _mm256_castps256_ps128(sumVec256);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    dotProduct = _mm_cvtss_f32(sum128);

#elif CLAWMIND_HAS_SSE
    __m128 sumVecSSE = _mm_setzero_ps();

    for (; index + 4 <= dimension; index += 4) {
        __m128 chunkA = _mm_loadu_ps(&vectorA[index]);
        __m128 chunkB = _mm_loadu_ps(&vectorB[index]);
        sumVecSSE = _mm_add_ps(sumVecSSE, _mm_mul_ps(chunkA, chunkB));
    }

    sumVecSSE = _mm_hadd_ps(sumVecSSE, sumVecSSE);
    sumVecSSE = _mm_hadd_ps(sumVecSSE, sumVecSSE);
    dotProduct = _mm_cvtss_f32(sumVecSSE);

#elif CLAWMIND_HAS_NEON
    float32x4_t sumVec = vdupq_n_f32(0.0f);

    for (; index + 4 <= dimension; index += 4) {
        float32x4_t chunkA = vld1q_f32(&vectorA[index]);
        float32x4_t chunkB = vld1q_f32(&vectorB[index]);
        sumVec = vfmaq_f32(sumVec, chunkA, chunkB);
    }

    dotProduct = vaddvq_f32(sumVec);
#endif

    for (; index < dimension; ++index) {
        dotProduct += vectorA[index] * vectorB[index];
    }

    return dotProduct;
}

/// Normalize a vector in-place to unit length.
/// Returns false if the vector has zero magnitude.
inline bool normalizeVector(std::span<float> vector) {
    float normSquared = 0.0f;
    for (float value : vector) {
        normSquared += value * value;
    }

    if (normSquared < 1e-10f) {
        return false;
    }

    float inverseNorm = 1.0f / std::sqrt(normSquared);
    for (float& value : vector) {
        value *= inverseNorm;
    }

    return true;
}

}  // namespace amind

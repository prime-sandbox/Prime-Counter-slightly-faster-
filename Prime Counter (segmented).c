#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define BASE_SEG (1155 * 16)

/*
 * High-performance segmented prime counting up to 1e10.
 *
 * Features:
 * - Odd-only representation (1 bit per odd number)
 * - Segmented sieve for large ranges
 * - Pre-sieved template for small primes (3,5,7,11)
 * - Manual workload balancing across threads
 * - Heavy loop unrolling and popcount-based counting
 *
 * This version prioritizes total throughput over repeated passes.
 */

int main() {

    const uint64_t limit = 10000000000ULL;
    const uint64_t half_limit = limit / 2;   // odd-only index space
    const uint64_t sqrt_limit = (uint64_t)sqrt((long double)limit);

    double start_t = omp_get_wtime();

    /* ------------------------------------------------------------
     * 1. Generate base prime list up to sqrt(limit)
     *    Only odd primes >= 13 are stored.
     * ------------------------------------------------------------ */

    uint8_t *small = (uint8_t *)calloc(sqrt_limit + 1, 1);

    for (uint64_t p = 3; p * p <= sqrt_limit; p += 2)
        if (!small[p])
            for (uint64_t i = p * p; i <= sqrt_limit; i += p << 1)
                small[i] = 1;

    uint32_t *primes = (uint32_t *)malloc(sizeof(uint32_t) * (sqrt_limit / 4));
    uint64_t p_cnt = 0;

    // Skip 3,5,7,11 (handled by template)
    for (uint64_t p = 13; p <= sqrt_limit; p += 2)
        if (!small[p])
            primes[p_cnt++] = (uint32_t)p;

    free(small);

    /* ------------------------------------------------------------
     * 2. Build pre-sieve template for primes {3,5,7,11}
     *
     * Template marks composite bits for these primes,
     * reused for every segment.
     * ------------------------------------------------------------ */

    uint8_t tpl[1155];
    memset(tpl, 0, 1155);

    uint32_t ps[] = {3, 5, 7, 11};

    for (int k = 0; k < 4; k++) {
        uint32_t p = ps[k];
        for (uint64_t j = (p - 1) / 2; j < 1155 * 8; j += p)
            tpl[j >> 3] |= (1 << (j & 7));
    }

    // Initial count includes known primes: 2,3,5,7,11
    uint64_t final_total = 5;

    static const uint8_t masks[8] = {1,2,4,8,16,32,64,128};

    /* ------------------------------------------------------------
     * 3. Parallel segmented sieve
     *
     * Each thread:
     * - Receives a weighted subrange
     * - Performs segmented sieving
     * - Counts primes using popcount
     * ------------------------------------------------------------ */

#pragma omp parallel num_threads(8)
    {
        uint64_t local_total = 0;
        int tid = omp_get_thread_num();

        /* --------------------------------------------------------
         * Manual load balancing
         *
         * Early ranges are heavier due to denser composites.
         * Weights compensate for uneven workload.
         * -------------------------------------------------------- */

        double weight[8] = {1.38, 1.38, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        double total_w = 8.76;

        double cum_w = 0;
        for (int i = 0; i < tid; i++)
            cum_w += weight[i];

        // Maintain segment alignment to preserve correctness
        uint64_t step = 1155 * 8;

        uint64_t my_low_bit =
            (((uint64_t)(half_limit * (cum_w / total_w)) / step) * step);

        uint64_t my_high_bit =
            (tid == 7)
            ? half_limit
            : (((uint64_t)(half_limit * ((cum_w + weight[tid]) / total_w)) / step) * step);

        /* --------------------------------------------------------
         * Segment size tuning
         *
         * First threads process denser regions, so they get
         * larger segments to amortize overhead.
         * -------------------------------------------------------- */

        uint32_t seg_size = (tid < 2) ? BASE_SEG * 2 : BASE_SEG;

        uint8_t *segment = (uint8_t *)malloc(seg_size + 64);
        uint64_t *next_idx = (uint64_t *)malloc(sizeof(uint64_t) * p_cnt);

        /* --------------------------------------------------------
         * Initialize next composite index per prime
         *
         * Ensures each segment starts marking at the correct offset.
         * -------------------------------------------------------- */

        for (uint64_t i = 0; i < p_cnt; i++) {
            uint64_t p = primes[i];
            uint64_t s = (p * p - 1) >> 1;

            if (s < my_low_bit) {
                uint64_t rem = (my_low_bit - s) % p;
                next_idx[i] = (rem == 0) ? my_low_bit : my_low_bit + (p - rem);
            } else {
                next_idx[i] = s;
            }
        }

        /* --------------------------------------------------------
         * Main segmented sieve loop
         * -------------------------------------------------------- */

        for (uint64_t low = my_low_bit; low < my_high_bit; low += seg_size * 8) {

            uint64_t high =
                (low + seg_size * 8 < my_high_bit)
                ? low + seg_size * 8
                : my_high_bit;

            uint32_t target_bytes = (uint32_t)((high - low + 7) >> 3);

            // Copy template across the segment
            for (uint32_t i = 0; i < target_bytes; i += 1155) {
                uint32_t len = (target_bytes - i < 1155) ? target_bytes - i : 1155;
                memcpy(segment + i, tpl, len);
            }

            /* ----------------------------------------------------
             * Mark composites using base primes
             * 8x loop unrolling for speed
             * ---------------------------------------------------- */

            for (uint64_t i = 0; i < p_cnt; i++) {
                uint64_t p = primes[i];
                uint64_t j = next_idx[i];
                uint64_t p8 = p << 3;

                while (j + p8 < high) {
                    segment[(j - low) >> 3] |= masks[j & 7]; j += p;
                    segment[(j - low) >> 3] |= masks[j & 7]; j += p;
                    segment[(j - low) >> 3] |= masks[j & 7]; j += p;
                    segment[(j - low) >> 3] |= masks[j & 7]; j += p;
                    segment[(j - low) >> 3] |= masks[j & 7]; j += p;
                    segment[(j - low) >> 3] |= masks[j & 7]; j += p;
                    segment[(j - low) >> 3] |= masks[j & 7]; j += p;
                    segment[(j - low) >> 3] |= masks[j & 7]; j += p;
                }

                while (j < high) {
                    segment[(j - low) >> 3] |= masks[j & 7];
                    j += p;
                }

                next_idx[i] = j;
            }

            // Handle number 1 explicitly
            if (low == 0)
                segment[0] |= 1;

            // Mask unused tail bits in the last byte
            uint64_t last_bit = high - low;
            if (last_bit < (uint64_t)target_bytes * 8) {
                for (uint64_t k = last_bit; k < (uint64_t)target_bytes * 8; k++)
                    segment[k >> 3] |= masks[k & 7];
            }

            /* ----------------------------------------------------
             * Count primes using popcount
             * ---------------------------------------------------- */

            uint64_t *seg64 = (uint64_t *)segment;
            uint32_t n64 = target_bytes >> 3;

            for (uint32_t k = 0; k < n64; k++)
                local_total += __builtin_popcountll(~seg64[k]);

            for (uint32_t k = n64 * 8; k < target_bytes; k++)
                local_total += __builtin_popcount((uint8_t)~segment[k]);
        }

#pragma omp atomic
        final_total += local_total;

        free(segment);
        free(next_idx);
    }

    /* ------------------------------------------------------------
     * 4. Final output
     * ------------------------------------------------------------ */

    printf("\n--- Enter a plausible name here ---\n");
    printf("Primes found: %llu (Expected: 455052511)\n",
           (unsigned long long)final_total);
    printf("Elapsed time: %.4f sec\n", omp_get_wtime() - start_t);

    free(primes);
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define LIMIT 1000000
#define HALF_LIMIT (LIMIT / 2)     // We store only odd numbers: n = 2*i + 1
#define SQRT_LIMIT 1000            // sqrt(1e6)
#define TPL_SIZE 1155              // Template size in bytes (pre-sieved pattern)

/*
 * High-performance prime counting benchmark.
 *
 * - Counts primes up to 1,000,000
 * - Uses odd-only bitset (1 bit per odd number)
 * - Pre-sieves small primes (3,5,7,11) via template copying
 * - Uses OpenMP for parallel execution
 * - Measures how many full sieve passes can be completed in 5 seconds
 */

int main() {

    /* ------------------------------------------------------------
     * 1. Generate small primes (odd primes >= 13 and <= 1000)
     *    These primes are used to mark composites in the main sieve.
     * ------------------------------------------------------------ */

    uint8_t small[SQRT_LIMIT + 1] = {0};

    // Simple sieve for odd numbers up to SQRT_LIMIT
    for (int p = 3; p * p <= SQRT_LIMIT; p += 2)
        if (!small[p])
            for (int i = p * p; i <= SQRT_LIMIT; i += p << 1)
                small[i] = 1;

    uint32_t primes[200];
    int p_cnt = 0;

    // Collect primes starting from 13
    // (3,5,7,11 are handled separately by the template)
    for (int p = 13; p <= SQRT_LIMIT; p += 2)
        if (!small[p])
            primes[p_cnt++] = p;

    /* ------------------------------------------------------------
     * 2. Build pre-sieve template for primes {3,5,7,11}
     *
     * The template marks composite positions for these small primes.
     * It will be copied repeatedly into the main sieve buffer,
     * avoiding redundant work every pass.
     * ------------------------------------------------------------ */

    uint8_t tpl[TPL_SIZE] = {0};
    uint32_t ps_tpl[] = {3, 5, 7, 11};

    for (int k = 0; k < 4; k++) {
        uint32_t p = ps_tpl[k];

        // Map odd number n = 2*j + 1  --> index j
        for (uint64_t j = (p - 1) / 2; j < TPL_SIZE * 8; j += p)
            tpl[j >> 3] |= (1 << (j & 7));
    }

    // Bit masks for individual bits inside a byte
    const uint8_t masks[8] = {1,2,4,8,16,32,64,128};

    uint64_t global_passes = 0;
    double start_t = omp_get_wtime();

    /* ------------------------------------------------------------
     * 3. Parallel benchmark loop
     *
     * Each thread:
     * - Allocates its own sieve buffer
     * - Repeats full sieving passes for ~5 seconds
     * - Counts how many passes it completes
     * ------------------------------------------------------------ */

#pragma omp parallel
    {
        // Sieve buffer:
        // 125000 bytes * 8 bits = 1,000,000 odd slots
        uint8_t *segment = (uint8_t *)malloc(125000 + 64);

        uint64_t local_passes = 0;

        while ((omp_get_wtime() - start_t) < 5.0) {

            /* ----------------------------------------------------
             * Copy pre-sieved template across the entire buffer
             * ---------------------------------------------------- */
            for (uint32_t i = 0; i < 125000; i += TPL_SIZE) {
                uint32_t len = (125000 - i < TPL_SIZE)
                               ? 125000 - i
                               : TPL_SIZE;
                memcpy(segment + i, tpl, len);
            }

            // Index 0 corresponds to number 1, which is not prime
            segment[0] |= 1;

            /* ----------------------------------------------------
             * Main sieving using primes >= 13
             *
             * Loop unrolling is used for speed.
             * j represents the index of (p*p), mapped to odd-only space.
             * ---------------------------------------------------- */
            for (int i = 0; i < p_cnt; i++) {
                uint32_t p = primes[i];
                uint32_t j = (p * p - 1) >> 1;
                uint32_t p8 = p << 3;

                // Unrolled loop (8 steps per iteration)
                while (j + p8 < HALF_LIMIT) {
                    segment[j >> 3] |= masks[j & 7]; j += p;
                    segment[j >> 3] |= masks[j & 7]; j += p;
                    segment[j >> 3] |= masks[j & 7]; j += p;
                    segment[j >> 3] |= masks[j & 7]; j += p;
                    segment[j >> 3] |= masks[j & 7]; j += p;
                    segment[j >> 3] |= masks[j & 7]; j += p;
                    segment[j >> 3] |= masks[j & 7]; j += p;
                    segment[j >> 3] |= masks[j & 7]; j += p;
                }

                // Tail handling
                while (j < HALF_LIMIT) {
                    segment[j >> 3] |= masks[j & 7];
                    j += p;
                }
            }

            local_passes++;
        }

        // Accumulate pass count across threads
#pragma omp atomic
        global_passes += local_passes;

        /* --------------------------------------------------------
         * Verification step (executed by master thread only)
         * -------------------------------------------------------- */
#pragma omp master
        {
            // Initial count includes known primes: 2,3,5,7,11
            uint64_t count = 5;

            for (uint32_t i = 0; i < HALF_LIMIT; i++) {
                // If bit is NOT set, the odd number is prime
                if (!(segment[i >> 3] & masks[i & 7]))
                    count++;
            }

            printf("Verification: Found %llu primes up to 1,000,000.\n",
                   (unsigned long long)count);

            if (count == 78498)
                printf("Status: SUCCESS (Faithful)\n");
            else
                printf("Status: FAILED\n");
        }

        free(segment);
    }

    /* ------------------------------------------------------------
     * 4. Final statistics
     * ------------------------------------------------------------ */

    double duration = omp_get_wtime() - start_t;

    printf("Passes: %llu\n", (unsigned long long)global_passes);
    printf("Time: %.4f sec\n", duration);
    printf("Avg: %.4f ms/pass\n",
           (duration * 1000.0) / global_passes);

    return 0;
}

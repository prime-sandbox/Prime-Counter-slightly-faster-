# Prime Counter (slightly faster)

High-performance prime counting programs written in C.

## Programs
- prime_counter_benchmark.c  
  Counts primes up to 1,000,000 within a 5-second time limit.

- prime_counter_exact.c  
  Counts the exact number of primes up to a given limit.

## Build
- gcc -O3 prime_counter_benchmark.c -o prime_benchmark
- gcc -O3 prime_counter_exact.c -o prime_exact

## Notes
This code prioritizes speed and low-level optimization over readability.

## Platform Notes
This code was primarily optimized for Snapdragon 480 / 4 Gen 2 class SoCs.
Performance and behavior may vary on other architectures and may require tuning.

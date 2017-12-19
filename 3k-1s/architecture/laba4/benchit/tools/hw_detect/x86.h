/**
 * check if RDTSC instruction is available
 */
int has_rdtsc();

/**
 * measures latency of the timestampcounter
 * @return latency in cycles, -1 if not available
 */
int get_rdtsc_latency();

/**
 * certain CPUs feature TSCs that are influenced by the powermanagement
 * those TSCs cannot be used to measure time
 * @return 1 if a usable TSC exists; 0 if no TSC is available or TSC is not usable
 */
int has_invariant_rdtsc();


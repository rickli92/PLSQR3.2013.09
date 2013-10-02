#include <mpi.h>
#include <memory.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int get_memory_usage_kb(long* vmrss_kb, long* vmsize_kb);
int get_cluster_memory_usage_kb(long* vmrss_per_process, long* vmsize_per_process, int root, int np);
int get_global_memory_usage_kb(long* global_vmrss, long* global_vmsize, int np);

void check_memory(int id, int np);

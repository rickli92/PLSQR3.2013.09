#ifndef VECOPS_H_

#define VECOPS_H_

// Some prototypes for vector operations

int min_vec(int *array, int len);
int max_vec(int *array, int len);
int sum_vec(int *array, int len);

void min_vec1(const int * array, const int len, int * min_idx, int * min_val);
void max_vec1(const int * array, const int len, int * max_idx, int * max_val);

#endif

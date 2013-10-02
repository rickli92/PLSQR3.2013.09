#include "vecops.h"

/*return the minimal element of a vector*/
int min_vec(int * array, int len)
{
        int min=array[0];
        int i;
        for(i=0; i<len; i++){
                if(array[i]<min)
                        min=array[i];
        }
        return min;
}

/*return the maximal element of a vector*/
int max_vec(int * array, int len)
{
        int max=array[0];
        int i;
        for(i=0; i<len; i++){
                if(array[i]>max)
                        max=array[i];
        }
        return max;
}

/*return the sum of all elements in a vector*/
int sum_vec(int * array, int len)
{
        int sum;
        int i;
        sum = 0;
        for(i=0; i<len; i++){
		sum = sum + array[i];
        }
        return sum;
}


/*return the minimal element with its index of a vector
 * IN: const int * array, const int len
 * OUT: pointer: int * index, int * max
 * */
void min_vec1(const int * array, const int len, int * min_idx, int * min_val)
{
        *min_val=array[0];
        *min_idx=0;
        int i;
        for(i=0; i<len; i++){
                if(array[i]<*min_val){
                        *min_val=array[i];
                        *min_idx=i;
                }
        }
}


void max_vec1(const int * array, const int len, int * max_idx, int * max_val)
{
        *max_val=array[0];
        *max_idx=0;
        int i;
        for(i=0; i<len; i++){
                if(array[i]>*max_val){
                        *max_val=array[i];
                        *max_idx=i;
                }
        }
}


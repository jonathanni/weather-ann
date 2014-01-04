#include <stdio.h>
#include <stdlib.h>

int * arr = 0;

int main()
{
  arr = (int *) calloc(6, sizeof(int));
  int addends[] = {1, 2, 3, 4, 5, 6};
  arr[0] = 3;
  int i = 0;

  for(i = 1; i < 6; i++)
    arr[i] = arr[i - 1] + addends[i - 1];

  for(i = 0; i < 6; i++)
    printf("%d\n", arr[i]);

  free(arr);
  return 0;
}

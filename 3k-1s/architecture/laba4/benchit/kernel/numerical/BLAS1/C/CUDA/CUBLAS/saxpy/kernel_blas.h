#include <cublas.h>

#define CUBLAS_CHECK(cmd) {cublasStatus error = cmd; if(error!=CUBLAS_STATUS_SUCCESS){printf("<%s>:%i error = %i\n",__FILE__,__LINE__, error); return 1;}}
#define CHECK_NULL(op) {if (NULL == op){printf("<%s>:%i operand was NULL\n",__FILE__,__LINE__); return 1;}}

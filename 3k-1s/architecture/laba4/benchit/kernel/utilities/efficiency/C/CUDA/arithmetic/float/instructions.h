/*
This is based on:
Demystifying GPU Microarchitecture through
Microbenchmarking
Henry Wong, Misel-Myrto Papadopoulou, Maryam Sadooghi-Alvandi, and Andreas Moshovos
Department of Electrical and Computer Engineering, University of Toronto
fhenry, myrto, alvandim, moshovosg@eecg.utoronto.ca
*/

typedef short		SHORT;
typedef	unsigned int 	UINT;
typedef int		INT;
typedef float		FLOAT;
typedef double 		DOUBLE;

#ifdef USE_INT
	#define TYPE INT
#endif
#ifdef USE_UINT
	#define TYPE UINT
#endif
#ifdef USE_FLOAT
	#define TYPE FLOAT
#endif
#ifdef USE_DOUBLE
	#define TYPE DOUBLE
#endif

#undef DEV_13
//#define DEV_13

/* WARP INTRINSICS */
#ifdef DEV_13
#define ALL(a, b)	a=__all(a==b)
#define ANY(a, b)	a=__any(a==b)
#else
#define ALL(a, b)	
#define ANY(a, b)	
#endif
/* #define SYNC(a, b)	__syncthreads() */

/* ARITHMETIC INSTRUCTIONS */
#define ADD(a, b) a+=b		
#define SUB(a, b)	a-=b	
#define MUL(a, b)	a*=b		
#define MAD(a, b)	a=a*b+a
#define DIV(a, b)	a/=b				
#define REM(a, b)	a%=b			
#define ABS(a, b)	a+=abs(b)		
#define NEG(a, b)	a^=-b		
#define MIN(a, b)	a=min(a+b,b)	
#define MAX(a, b)	a=max(a+b,b)	

#define ADD2(a, b, c) a+=b
#define SUB2(a, b, c)	a-=b
#define MUL2(a, b, c)	a*=b
#define MAD2(a, b, c)	a=b-a*c
#define DIV2(a, b, c)	a/=b

/* LOGIC INSTRUCTIONS */
#define AND(a, b)	a&=b				
#define OR(a, b)	a|=b				
#define XOR(a, b)	a^=b				
#define SHL(a, b)	a<<=b			
#define SHR(a, b)	a>>=b			

/* TO BE TESTED LATER */
#define NOT(a, b)	a=~b				
#define NOT2(a, b)	if (a>=b) a=~b				
#define CNOT(a, b)	a^=(b==0)?1:0

/* TO BE TESTED LATER */
#define ANDNOT(a, b)	a&=~b
#define ORNOT(a, b)	a|=~b
#define XORNOT(a, b)	a^=~b
#define ADDNOT(a, b) 	a+=~b
#define ANDNOTNOT(a, b)	a=~a&~b

/* ARITHMETIC INSTRINSICS: INTEGER */
#define MUL24(a, b)	a=__mul24(a, b)
#define UMUL24(a, b)	a=__umul24(a, b)
#define MULHI(a, b)	a=__mulhi(a, b)
#define UMULHI(a, b)	a=__umulhi(a, b)
#define SAD(a, b)	a=__sad(a, b, a)	
#define USAD(a, b)	a=__usad(a, b, a)	

/* ARITHMETIC INTRINSICS: FLOAT */
#define FADD_RN(a, b)	a=__fadd_rn(a, b)
#define FADD_RZ(a, b)	a=__fadd_rz(a, b)
#define FMUL_RN(a, b)	a=__fmul_rn(a, b)
#define FMUL_RZ(a, b)	a=__fmul_rz(a, b)
#define FDIVIDEF(a, b)	a=__fdividef(a, b)

/* ARITHMETIC INTRINSICS: DOUBLE. Requires SM1.3 */
#ifdef DEV_13
#define DADD_RN(a, b)	a=__dadd_rn(a, b)
#else
#define DADD_RN(a, b)
#endif

/* MATH INSTRUCTIONS: FLOAT */
#define RCP(a, b)	a+=1/b
#define SQRT(a, b)	a=sqrt(b)
#define RSQRT(a, b)	a=rsqrt(b)

#define SIN(a, b)	a=sinf(b)
#define COS(a, b)	a=cosf(b)
#define TAN(a, b)	a=tanf(b)
#define EXP(a, b)	a=expf(b)
#define EXP10(a, b)	a=exp10f(b)
#define LOG(a, b)	a=logf(b)
#define LOG2(a, b)	a=log2f(b)
#define LOG10(a, b)	a=log10f(b)
#define POW(a, b)	a=powf(a, b)


/* MATH INTRINSICS: FLOAT */
#define SINF(a, b)	a=__sinf(b)
#define COSF(a, b)	a=__cosf(b)
//#define SINCOSF
#define TANF(a, b)	a=__tanf(b)
#define EXPF(a, b)	a=__expf(b)
#define EXP2F(a, b)	a=exp2f(b)
#define EXP10F(a, b)	a=__exp10f(b)
#define LOGF(a, b)	a=__logf(b)
#define LOG2F(a, b)	a=__log2f(b)
#define LOG10F(a, b)	a=__log10f(b)
#define POWF(a, b)	a=__powf(a, b)

/* CONVERSION INTRINSICS */
#define INTASFLOAT(a, b)		a=__int_as_float(b)
#define FLOATASINT(a, b)		a=__float_as_int(b)

/* MISC INTRINSICS */
#define POPC(a, b)	a=__popc(b)
#define SATURATE(a, b)	a=saturate(b)
#define CLZ(a, b)	a=__clz(b)  //count leading zeros	
#define CLZLL(a, b)	a=__clzll(b)  //count leading zeros	
#define FFS(a, b)	a=__ffs(b)
#define FFSLL(a, b)	a=__ffsll(b)

/* DATA MOVEMENT AND CONVERSION INSTRUCTIONS */
#define MOV(a, b)	a+=b; b=a 
#define MOV4(a, b, c)	a=b^c; b=a

#define IF(a, b)	if(a == b) a^=b

/* ATOMIC INSTRUCTIONS */
#define ATOMICADD(a,b)	atomicAdd(a, b)

__device__ uint get_smid(){
	uint ret;
	asm("mov.u32 %0, %smid;" : "=r"(ret) );
	return ret;
}

#define K_OP_DEP(OP, DEP, TYPE)\
extern "C"\
__global__ void K_##OP##_##TYPE##_DEP##DEP (int numThreads, unsigned int *ts, unsigned int* out, TYPE p1, TYPE p2, int its) 	\
{														\
	register TYPE t1 = p1;												\
	register TYPE t2 = p2;												\
	uint smid, smid2;\
	unsigned int start_time=0, stop_time=1;									\
	int id = blockIdx.x*blockDim.x + threadIdx.x; \
	if(id >= numThreads) return; \
														\
		for (int i=0;i<its;i++){												\
			smid = get_smid();\
			start_time = clock();									\
			repeat##DEP(OP(t1, t2); OP(t2, t1);)							\
			stop_time = clock();									\
			smid2 = get_smid();\
		}												\
														\
	if(t1==0) out[0] = (unsigned int )(t1 + t2);									\
	if(id % 32 == 0){\
		ts[id/32 * 2] = start_time;						\
		ts[id/32 * 2 + 1] = stop_time;						\
		out[id/32+1] = (smid==smid2)?smid+1:smid2+1+1000*(smid+1);\
	}\
}

#define _OP2(OP, i, t1, t2, t3) OP##i(t1,t2,t3)
#define OP2(OP, t1, t2, t3) _OP2(OP, 2,t1,t2,t3)

#define K_OP2_DEP(OP, DEP, TYPE)\
extern "C"\
__global__ void K_##OP##_##TYPE##_DEP##DEP (int numThreads, unsigned int *ts, unsigned int* out, TYPE p1, TYPE p2, int its) 	\
{														\
	register TYPE t1 = p1;												\
	register TYPE t2 = p2;												\
	register TYPE t3 = p1+1;												\
	register TYPE t4 = p2+3;												\
	uint smid, smid2;\
	unsigned int start_time=0, stop_time=1;									\
	int id = blockIdx.x*blockDim.x + threadIdx.x; \
	if(id >= numThreads) return; \
														\
		for (int i=0;i<its;i++){												\
			smid = get_smid();\
			start_time = clock();									\
			repeat##DEP(OP2(OP, t1, p1, p2); OP2(OP, t2, p1, p2); OP2(OP, t3, p1, p2); OP2(OP, t4, p1, p2);)							\
			stop_time = clock();									\
			smid2 = get_smid();\
		}												\
														\
	if(t1==0) out[0] = (unsigned int )(t1 + t2 + t3 + t4);									\
	if(id % 32 == 0){\
		ts[id/32 * 2] = start_time;						\
		ts[id/32 * 2 + 1] = stop_time;						\
		out[id/32+1] = (smid==smid2)?smid+1:smid2+1+1000*(smid+1);\
	}\
}

/*	smids[threadIdx.x] = (smid == smid2)?smid:0xDEAD;\
	__syncthreads();									\
	if(threadIdx.x==0){\
		for(int i=0;i<blockDim.x && id+i<numThreads;i++)\
			if(smids[i]!=smid) out[id+1] = 0xDEAD;\
	}\
*/
#ifdef USE_UINT
/* WARP VOTE INTRINSICS -- NEEDS SM1.2 */
K_OP_DEP(ALL, 128, UINT)
K_OP_DEP(ANY, 128, UINT)

/* ARITHMETIC INSTRUCTIONS: UINT*/
K_OP_DEP(ADD, 128, UINT)
K_OP_DEP(SUB, 128, UINT)
K_OP_DEP(MUL, 128, UINT)
K_OP_DEP(DIV, 128, UINT)
K_OP_DEP(REM, 128, UINT)
K_OP_DEP(MAD, 128, UINT)
K_OP_DEP(MIN, 128, UINT)
K_OP_DEP(MAX, 128, UINT)
#endif

#ifdef USE_INT
/* ARITHMETIC INSTRUCTIONS: INT */
K_OP_DEP(ADD, 128, INT)
K_OP_DEP(SUB, 128, INT)
K_OP_DEP(MUL, 128, INT)
K_OP_DEP(DIV, 128, INT)
K_OP_DEP(REM, 128, INT)
K_OP_DEP(MAD, 128, INT)
K_OP_DEP(ABS, 128, INT)
K_OP_DEP(NEG, 128, INT)
K_OP_DEP(MIN, 128, INT)
K_OP_DEP(MAX, 128, INT)
#endif

#ifdef USE_FLOAT
/* ARITHMETIC INSTRUCTIONS: FLOAT */
K_OP_DEP(ADD, 128, FLOAT)
K_OP_DEP(SUB, 128, FLOAT)
K_OP_DEP(MUL, 128, FLOAT)
K_OP_DEP(DIV, 128, FLOAT)
K_OP_DEP(MAD, 128, FLOAT)
K_OP_DEP(ABS, 128, FLOAT)
K_OP_DEP(MIN, 128, FLOAT)
K_OP_DEP(MAX, 128, FLOAT)
#endif

#ifdef USE_DOUBLE
/* ARITHMETIC INSTRUCTIONS: DOUBLE */
K_OP_DEP(ADD, 128, DOUBLE)
K_OP_DEP(SUB, 128, DOUBLE)
K_OP_DEP(MUL, 128, DOUBLE)
K_OP_DEP(DIV, 128, DOUBLE)
K_OP_DEP(MAD, 128, DOUBLE)
K_OP_DEP(ABS, 128, DOUBLE)
K_OP_DEP(MIN, 128, DOUBLE)
K_OP_DEP(MAX, 128, DOUBLE)
#endif

/* LOGIC INSTRUCTIONS */
#ifdef USE_UINT
K_OP_DEP(AND,  128, UINT) 
K_OP_DEP(OR,   128, UINT) 
K_OP_DEP(XOR,  128, UINT) 
K_OP_DEP(SHL,  128, UINT) 
K_OP_DEP(SHR,  128, UINT) 

K_OP_DEP(NOT,  128, UINT) 
K_OP_DEP(CNOT, 128, UINT) 

K_OP_DEP(ANDNOT,  128, UINT) 
K_OP_DEP(ORNOT,   128, UINT) 
K_OP_DEP(XORNOT,  128, UINT) 
K_OP_DEP(ADDNOT,  128, UINT) 
K_OP_DEP(ANDNOTNOT,  128, UINT) 

/* ARITHMETIC INSTRINSICS: UINT/INT */
K_OP_DEP(UMUL24, 128, UINT)
K_OP_DEP(UMULHI, 128, UINT)
K_OP_DEP(USAD, 128, UINT)
#endif

#ifdef USE_INT
K_OP_DEP(MUL24, 128, INT)
K_OP_DEP(MULHI, 128, INT)
K_OP_DEP(SAD, 128, INT)
K_OP_DEP(NOT2,  128, INT) 
#endif

#ifdef USE_FLOAT
/* ARITHMETIC INSTRINSICS: FLOAT */
K_OP_DEP(FADD_RN, 128, FLOAT)
K_OP_DEP(FADD_RZ, 128, FLOAT)
K_OP_DEP(FMUL_RN, 128, FLOAT)
K_OP_DEP(FMUL_RZ, 128, FLOAT)
K_OP_DEP(FDIVIDEF, 128, FLOAT)

/* MATH INSTRUCTIONS: FLOAT */
K_OP_DEP(RCP, 128, FLOAT)
//K_OP_DEP(SQRT, 128, FLOAT)
//K_OP_DEP(RSQRT, 128, FLOAT)
/*
K_OP_DEP(SIN, 128, FLOAT)
K_OP_DEP(COS, 128, FLOAT)
K_OP_DEP(TAN, 128, FLOAT)
K_OP_DEP(EXP, 128, FLOAT)
K_OP_DEP(EXP10, 128, FLOAT)
K_OP_DEP(LOG, 128, FLOAT)
K_OP_DEP(LOG2, 128, FLOAT)
K_OP_DEP(LOG10, 128, FLOAT)
K_OP_DEP(POW, 128, FLOAT)
*/

/* MATH INTRINSICS: FLOAT */
K_OP_DEP(SINF, 128, FLOAT)
K_OP_DEP(COSF, 128, FLOAT)
K_OP_DEP(TANF, 128, FLOAT)
K_OP_DEP(EXPF, 128, FLOAT)
K_OP_DEP(EXP2F, 128, FLOAT)
K_OP_DEP(EXP10F, 128, FLOAT)
K_OP_DEP(LOGF, 128, FLOAT)
K_OP_DEP(LOG2F, 128, FLOAT)
K_OP_DEP(LOG10F, 128, FLOAT)
K_OP_DEP(POWF, 128, FLOAT)
#endif

/* INSTRINSICS: DOUBLE */
K_OP_DEP(DADD_RN, 128, DOUBLE)

/* CONVERSION */
K_OP_DEP(INTASFLOAT, 128, UINT)
K_OP_DEP(FLOATASINT, 128, FLOAT)

/* MISC */
K_OP_DEP(POPC, 128, UINT)
K_OP_DEP(CLZ, 128, UINT)
K_OP_DEP(CLZLL, 128, UINT)
K_OP_DEP(FFS, 128, UINT)
K_OP_DEP(FFSLL, 128, UINT)
K_OP_DEP(SATURATE, 128, FLOAT)

/* ATOMIC INSTRUCTIONS NEEDS SM1.1 and SM1.2 for SHARED*/
//K_OP_DEP_ATOMIC(ATOMICADD, 128, UINT)

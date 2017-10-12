	.file	"laba1.c"
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC2:
	.string	"%lf\n"
.LC3:
	.string	"yes"
	.section	.text.unlikely,"ax",@progbits
.LCOLDB4:
	.section	.text.startup,"ax",@progbits
.LHOTB4:
	.p2align 4,,15
	.globl	main
	.type	main, @function
main:
.LFB39:
	.cfi_startproc
#APP
# 15 "rdtsc.h" 1
	rdtsc
# 0 "" 2
#NO_APP
	movsd	.LC0(%rip), %xmm0
	salq	$32, %rdx
	movl	%eax, %eax
	movq	%rdx, %rcx
	movapd	%xmm0, %xmm1
	orq	%rax, %rcx
	movl	$2000000000, %eax
	.p2align 4,,10
	.p2align 3
.L2:
	subq	$1, %rax
	addsd	%xmm0, %xmm1
	addsd	%xmm3, %xmm2
	addsd	%xmm4, %xmm5
	addsd	%xmm6, %xmm7
	addsd	%xmm8, %xmm9
	addsd	%xmm10, %xmm11
	addsd	%xmm12, %xmm13
	addsd	%xmm14, %xmm15
	jne	.L2
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
#APP
# 15 "rdtsc.h" 1
	rdtsc
# 0 "" 2
#NO_APP
	salq	$32, %rdx
	movl	%eax, %eax
	orq	%rax, %rdx
	subq	%rcx, %rdx
	js	.L3
	pxor	%xmm0, %xmm0
	cvtsi2sdq	%rdx, %xmm0
.L4:
	divsd	.LC1(%rip), %xmm0
	movl	$.LC2, %esi
	movl	$1, %edi
	movl	$1, %eax
	movsd	%xmm1, 8(%rsp)
	call	__printf_chk
	movsd	8(%rsp), %xmm1
	ucomisd	.LC1(%rip), %xmm1
	jbe	.L5
	movl	$.LC3, %edi
	call	puts
.L5:
	xorl	%eax, %eax
	addq	$24, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L3:
	.cfi_restore_state
	movq	%rdx, %rax
	pxor	%xmm0, %xmm0
	shrq	%rax
	andl	$1, %edx
	orq	%rdx, %rax
	cvtsi2sdq	%rax, %xmm0
	addsd	%xmm0, %xmm0
	jmp	.L4
	.cfi_endproc
.LFE39:
	.size	main, .-main
	.section	.text.unlikely
.LCOLDE4:
	.section	.text.startup
.LHOTE4:
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC0:
	.long	0
	.long	1072693248
	.align 8
.LC1:
	.long	0
	.long	1105055077
	.ident	"GCC: (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609"
	.section	.note.GNU-stack,"",@progbits

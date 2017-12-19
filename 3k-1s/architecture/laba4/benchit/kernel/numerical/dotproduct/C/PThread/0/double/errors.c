/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: errors.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/dotproduct/C/PThread/0/double/errors.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Core for dot product of two vectors with posix threads
 *******************************************************************/

#include "errors.h"

char errtxts[17][80] = { "no permission to create directory",
   "name already exist",
   "name outside accessible address space",
   "no write permission for process",
   "name too long",
   "part of name is invalid (does not exist or is a link)",
   "part of name is invalid (does not exist)",
   "insufficient kernel memory",
   "no access to read-only file",
   "too many symbolic links",
   "not enough space left on device",
   "i/o error occurred",
   "invalid file descriptor",
   "not enough resources to create an new thread",
   "no corresponding thread found",
   "unable to phtread_join to myself",
   "unknown error"
};

char *error_text(int errnum) {
   switch (errnum) {
      case EPERM:
         return errtxts[0];
      case EEXIST:
         return errtxts[1];
      case EFAULT:
         return errtxts[2];
      case EACCES:
         return errtxts[3];
      case ENAMETOOLONG:
         return errtxts[4];
      case ENOENT:
         return errtxts[5];
      case ENOTDIR:
         return errtxts[6];
      case ENOMEM:
         return errtxts[7];
      case EROFS:
         return errtxts[8];
      case ELOOP:
         return errtxts[9];
      case ENOSPC:
         return errtxts[10];
      case EIO:
         return errtxts[11];
      case EBADF:
         return errtxts[12];
      case EAGAIN:
         return errtxts[13];
      case ESRCH:
         return errtxts[14];
      case EDEADLK:
         return errtxts[15];
      default:
         return errtxts[16];
   }
}


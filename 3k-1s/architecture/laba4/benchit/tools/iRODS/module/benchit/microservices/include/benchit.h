#ifndef BENCHIT_H
#define BENCHIT_H

#include <sys/time.h>
#include "rods.h"
#include "reGlobalsExtern.h"
#include "rsGlobalExtern.h"
#include "rcGlobalExtern.h"

int msi_start_measurement(ruleExecInfo_t *rei );
int msi_inside_measurement(msParam_t *inpParam, ruleExecInfo_t *rei );
int msi_end_measurement(msParam_t *inpParam, ruleExecInfo_t *rei );

#endif	/* BENCHIT_H */

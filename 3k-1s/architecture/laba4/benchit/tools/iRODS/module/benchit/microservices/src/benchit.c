#include "benchit.h"
#define STRING_BUFFER 100
#define STRING_BUFFER_BIG 200


int msi_start_measurement(ruleExecInfo_t *rei)
{
	struct timeval zeit;
	RE_TEST_MACRO("    measurement - start");
   char mystr[STRING_BUFFER];
   gettimeofday(&zeit,0);
   double zeit_wert = (double) zeit.tv_sec + ((double) zeit.tv_usec / 1000000.0);
   snprintf(mystr, STRING_BUFFER, "\nbenchit\tstart\t%lf\n", zeit_wert);
   _writeString("stdout", mystr, rei);
	return 0;
}

int msi_inside_measurement(msParam_t *inpParam, ruleExecInfo_t *rei)
{
   char *str_value;
   if (strcmp (inpParam->type, STR_MS_T) == 0)
   {
      str_value = strdup((char *) inpParam->inOutStruct);
   }
   else
   {
      fprintf(stderr,"Error(msi_inside_measurement): No valid parameter"); 
   }
   struct timeval zeit;
	RE_TEST_MACRO("    measurement - inside");
   gettimeofday(&zeit,NULL);
   double zeit_wert = (double) zeit.tv_sec + ((double) zeit.tv_usec / 1000000.0);
   char mystr[STRING_BUFFER_BIG]; 
   snprintf(mystr, STRING_BUFFER_BIG, "\nbenchit\t%s\t%lf\n", str_value,zeit_wert);
   _writeString("stdout", mystr, rei);
      
	return 0;
}

int msi_end_measurement(msParam_t *inpParam, ruleExecInfo_t *rei)
{
   char *str_value;
   if (strcmp (inpParam->type, STR_MS_T) == 0)
   {
      str_value = strdup((char *) inpParam->inOutStruct);
   }
   else
   {
      fprintf(stderr,"Error(msi_end_measurement): No valid parameter"); 
   }
   struct timeval zeit;
	RE_TEST_MACRO("    measurement - end");
   gettimeofday(&zeit,NULL);
   double zeit_wert = (double) zeit.tv_sec + ((double) zeit.tv_usec / 1000000.0);
   char mystr[STRING_BUFFER_BIG]; 
   snprintf(mystr, STRING_BUFFER_BIG, "\nbenchit\t%s\t%lf\n", str_value, zeit_wert);
   _writeString("stdout", mystr, rei);

	return 0;
}

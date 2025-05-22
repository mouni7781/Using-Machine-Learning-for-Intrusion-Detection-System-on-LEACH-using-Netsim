/************************************************************************************
 * Copyright (C) 2020                                                               *
 * TETCOS, Bangalore. India                                                         *
 *                                                                                  *
 * Tetcos owns the intellectual property rights in the Product and its content.     *
 * The copying, redistribution, reselling or publication of any or all of the       *
 * Product or its content without express prior written consent of Tetcos is        *
 * prohibited. Ownership and / or any other right relating to the software and all  *
 * intellectual property rights therein shall remain at all times with Tetcos.      *
 *                                                                                  *
 * Author:    Shashi Kant Suman                                                     *
 *                                                                                  *
 * ---------------------------------------------------------------------------------*/
# include <math.h>
#include "main.h"
#include "802_15_4.h"


int lf_NP_RandomBackOffTime(int);
/** The backoff procedure, the STA shall set its Backoff Timer to a random backoff time
using the equation		
Backoff Time = Random() × aSlotTime.
All backoff slots occur following a DIFS period during which the medium is determined to be
idle for the duration of the DIFS period. 
*/
int fn_NetSim_Zigbee_BackoffTimeCalculation(int nBackoffExponent,double* dBackoffTime,int nUnitBacoffPeriod,METRICS** pstruMetrics)
{
	int nBackoffTimes;
	nBackoffTimes = (int)pow(2,nBackoffExponent);
	//nBackoffTimes = lf_NP_RandomBackOffTime(nBackoffTimes);
	nBackoffTimes = (int)(((unsigned long long int)fn_NetSim_Utilities_GenerateRandomNo(&ulBackoffSeed1,&ulBackoffSeed2))%nBackoffTimes);
	*dBackoffTime = nBackoffTimes*dUnitSymbolTime*nUnitBacoffPeriod;
	pstruMetrics[pstruEventDetails->nDeviceId-1]->pstruIEEE802_15_4_Metrics->dTotalbackofftime += (*dBackoffTime);
	pstruMetrics[pstruEventDetails->nDeviceId-1]->pstruIEEE802_15_4_Metrics->nNumberofBackoffCall++;
	return 1;
}
/** 
	This Function is used to generate the Random time based on seed values 
*/
int lf_NP_RandomBackOffTime(int nMaxRandBackoff)
{
  long lx = 0;
  long double ldy = 0;
  int ltemp;

  ulBackoffSeed1 = (unsigned long) ((40014 * ulBackoffSeed1)  % (unsigned long) (2147483563));
  ulBackoffSeed2 =  (unsigned long) ((40692 * ulBackoffSeed2) % (unsigned long) (2147483399));

  lx = (long) ((ulBackoffSeed1 - ulBackoffSeed2) % (long) (2147483562));

  if (lx != 0)
  {
	 ldy = (long double) ((long double) (lx) / (long double) (2147483563));
  }
  else
  {
	 ldy = (long double) ((long double) (2147483562) / (long double) (2147483563));
  }


  ltemp = (int)(ldy*100000000);
  ltemp = (ltemp%nMaxRandBackoff);
  return ltemp;
}

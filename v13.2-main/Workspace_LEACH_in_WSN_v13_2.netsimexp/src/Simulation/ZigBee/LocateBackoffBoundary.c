/************************************************************************************
 * Copyright (C) 2020                                                               *
 * TETCOS, Bangalore. India                                                         *
 *                                                                                  *
 * Tetcos owns the intellectual property rights in the Product and its content.     *
 * The copying, redistribution, reselling or publication of any or all of the       *
 * Product or its content without express prior written consent of Tetcos is        *
 * prohibited. Ownership and / or any other right relating to the software and all *
 * intellectual property rights therein shall remain at all times with Tetcos.      *
 *                                                                                  *
 * Author:    Shashi Kant Suman                                                       *
 *                                                                                  *
 * ---------------------------------------------------------------------------------*/
#include "main.h"
# include "802_15_4.h"
/** This function is used to locate the boundary of the next backoff slot */
int fn_NetSim_Zigbee_LocateBackoffBoundary(double dTime, double* dBoundaryTime, int aUnitBackoffPeriod, SUPERFRAME* pstruSuperFrame)
{
	int i =0;
	while(pstruSuperFrame->dSuperFrameStartTime + (double)i*aUnitBackoffPeriod*dUnitSymbolTime < dTime)
		i++;

	*dBoundaryTime = pstruSuperFrame->dSuperFrameStartTime + (double)i*aUnitBackoffPeriod*dUnitSymbolTime;
	if(*dBoundaryTime < pstruSuperFrame->dSuperFrameLength + pstruSuperFrame->dSuperFrameStartTime)
	{
		*dBoundaryTime = *dBoundaryTime;
	}
	else
		*dBoundaryTime = pstruSuperFrame->dSuperFrameStartTime + pstruSuperFrame->dSuperFrameLength+pstruSuperFrame->dTimeSlotLength;
	return 1;
}

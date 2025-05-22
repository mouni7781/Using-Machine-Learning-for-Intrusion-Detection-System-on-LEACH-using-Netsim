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
# include "main.h"
# include "802_15_4.h"
/**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The PHY shall provide the capability to perform CCA according to at least one of the following
three methods:
— CCA Mode 1: Energy above threshold. CCA shall report a busy medium upon detecting any
energy above the ED threshold.
— CCA Mode 2: Carrier sense only. CCA shall report a busy medium only upon the detection
of a signal compliant with this standard with the same modulation and spreading characteristics
of the PHY that is currently in use by the device. This signal may be above or below the ED
threshold.
— CCA Mode 3: Carrier sense with energy above threshold. CCA shall report a busy medium
using a logical combination of
— Detection of a signal with the modulation and spreading characteristics of this standard and
— Energy above the ED threshold, where the logical operator may be AND or OR.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
int fn_NetSim_Zigbee_CCA(double dInterferencePower, CCAMODE nCCA_Mode, double dReceiverSensivity, double dEDThreshold,METRICS** pstruMetrics)
{
	pstruMetrics[pstruEventDetails->nDeviceId-1]->pstruIEEE802_15_4_Metrics->nCCAAttempt++;
	switch(nCCA_Mode)
	{
	case CARRIERSENSE:
		if(dInterferencePower < dReceiverSensivity)
		{
			pstruMetrics[pstruEventDetails->nDeviceId-1]->pstruIEEE802_15_4_Metrics->nSuccessfulCCAAttempt++;
			//Channel is idle
			return CHANNEL_IDLE;
		}
		break;
	case ENERGYABOVETHERESHOLD:
		if(dInterferencePower < dEDThreshold)
		{
			pstruMetrics[pstruEventDetails->nDeviceId-1]->pstruIEEE802_15_4_Metrics->nSuccessfulCCAAttempt++;
			//Channel is idle
			return CHANNEL_IDLE;
		}
		break;
	case CARRIERSENSE_AND_ENERGYABOVETHERESHOLD:
		if(dInterferencePower < dEDThreshold && dInterferencePower < dReceiverSensivity)
		{
			pstruMetrics[pstruEventDetails->nDeviceId-1]->pstruIEEE802_15_4_Metrics->nSuccessfulCCAAttempt++;
			//Channel is idle
			return CHANNEL_IDLE;
		}
		break;
	case CARRIERSENSE_OR_ENERGYABOVETHERESHOLD:
		if(dInterferencePower < dEDThreshold || dInterferencePower < dReceiverSensivity)
		{
			pstruMetrics[pstruEventDetails->nDeviceId-1]->pstruIEEE802_15_4_Metrics->nSuccessfulCCAAttempt++;
			//Channel is idle
			return CHANNEL_IDLE;
		}
		break;
	default:
		printf("Unknown CCA mode\nTeminating Application....IEEE 802.15.4");
		exit(0);
		break;
	}
	pstruMetrics[pstruEventDetails->nDeviceId-1]->pstruIEEE802_15_4_Metrics->nFailedCCA++;
	//CCA fails
	return CHANNEL_BUSY;
}

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
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	The SINR is defined as the power of signal divided by the sum of the interference power and the 
	power of background noise. 
	SINR(dBm) = log( Receivedpower(mW) / (Interference Niose(mW) + Thermal Noise(mW))) 
	The logarithm mentioned above is to base 10
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
int fn_NetSim_Zigbee_CalculateSINR(double *SNR, double dTotalReceivedPower,double dReceivedPower)
{
	double dThermalNoisePower_1Hz = -174.0; //At room temp. In dBm
	double dThemalNoise; //in dBm
	double dBandwidth = 5000000.0; //IEEE 802.15.4 standard for 2.4 GHz.
	double dNoisePower = 0.0;
	double dSNR;
	//Calculate the thermal noise
	dThemalNoise = dThermalNoisePower_1Hz + 10 * log10(dBandwidth);
	
	dTotalReceivedPower = 10*log10(dTotalReceivedPower);
	dReceivedPower = 10*log10(dReceivedPower);

	dNoisePower = dTotalReceivedPower - dReceivedPower;
	
	//Calculate the SNR
	dSNR = dReceivedPower-dThemalNoise-dNoisePower;

	//SNR in db
	*SNR = dSNR;
	return 1;

}


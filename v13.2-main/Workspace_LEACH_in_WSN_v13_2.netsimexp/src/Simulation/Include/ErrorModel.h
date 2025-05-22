/************************************************************************************
* Copyright (C) 2020
* TETCOS, Bangalore. India															*

* Tetcos owns the intellectual property rights in the Product and its content.     *
* The copying, redistribution, reselling or publication of any or all of the       *
* Product or its content without express prior written consent of Tetcos is        *
* prohibited. Ownership and / or any other right relating to the software and all  *
* intellectual property rights therein shall remain at all times with Tetcos.      *
* Author:	Shashi Kant Suman														*
* ---------------------------------------------------------------------------------*/
#ifndef _NETSIM_ERROR_MODEL_H_
#define _NETSIM_ERROR_MODEL_H_

#include "Wireless.h"

typedef struct stru_ber
{
	double dSNR;
	double dBER;
}BER, *ptrBER;

_declspec(dllexport) double calculate_ber(double snr,BER ber_table[],size_t table_len);
_declspec(dllexport) double calculate_sinr(double dReceivedPower_dbm, double interferencePower_dBm, double bandwidth_mHz);
_declspec(dllexport) double calculate_BER(PHY_MODULATION modulation,
	double dReceivedPower_dBm,
	double dInterferencePower_dBm,
	double dBandwidth_MHz);

_declspec(dllexport) double propagation_calculateRXSensitivity(double PEP,
															   double refPacketSize /* In bytes*/,
															   PHY_MODULATION modulation,
															   double bandwidth);
_declspec(dllexport) double add_power_in_dbm(double p1_dbm, double p2_dbm);
_declspec(dllexport) double substract_power_in_dbm(double p1_dbm, double p2_dbm);
_declspec(dllexport) double find_power_from_snr(double snr,
												double bandwidth);
_declspec(dllexport) double find_snr_from_ber(double ber,
											  PHY_MODULATION modulation);

//New BER 
_declspec(dllexport) double Calculate_ber_by_calculation(double sinr, PHY_MODULATION modulation,
														 double dataRate_mbps, double bandwidth_mHz);
_declspec(dllexport) double calculate_rxpower_by_ber(double refBer, PHY_MODULATION modulation,
													 double datarate_mbps, double bandwidth_mhz);
_declspec(dllexport) double calculate_rxpower_by_per(double per, double refPacketSize,
													 PHY_MODULATION modulation,
													 double datarate_mbps, double bandwidth_mhz);

/**
	Used to read BER file.
	File format
		<SNR1>,<BER1>,
		<SNR2>,<BER2>,
		....
		....
		....
		<SNRn>,<BERn>,
	Input
		BER file name with relative path to IO path
		pointer to BER table length
	Return
		BER table of size len
	Note: SNR value in file must be in increasing order.
		  This API doesn't check this, so if it is not will
		  result wrong behaviour.
*/
_declspec(dllexport) ptrBER read_ber_file(char* file, size_t* len);

//FEC BER
_declspec(dllexport) double calculate_FEC_BER(PHY_MODULATION modulation,
	double codingRate /* 1/2, 2/3, 3/4, 5/6 */,
	double dReceivedPower_dBm,
	double interferencePower_dBm,
	double dBandwidth_MHz,
	double dataRate_mbps);

#endif //_NETSIM_ERROR_MODEL_H_

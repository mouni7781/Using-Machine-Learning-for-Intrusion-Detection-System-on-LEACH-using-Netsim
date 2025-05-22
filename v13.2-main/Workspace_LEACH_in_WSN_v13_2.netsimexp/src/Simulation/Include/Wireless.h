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
 * Author:    Shashi Kant Suman                                                      *
 *                                                                                  *
 * ---------------------------------------------------------------------------------*/
#ifndef _NETSIM_WIRELESS_H_
#define _NETSIM_WIRELESS_H_
#ifdef  __cplusplus
extern "C" {
#endif

#ifndef NETSIM_ID
	typedef unsigned int NETSIM_ID; //unique id
#endif

	typedef double CODINGRATE;

#include "PropagationModel.h"

	typedef enum enum_Modulation
	{
		Modulation_Zero, // Keep me in first
		Modulation_DBPSK,
		Modulation_DQPSK,
		Modulation_CCK5_5,
		Modulation_CCK11,
		Modulation_BPSK,
		Modulation_QPSK,
		Modulation_8_QAM,
		Modulation_16_QAM,
		Modulation_32_QAM,
		Modulation_64_QAM,
		Modulation_128_QAM,
		Modulation_256_QAM,
		Modulation_GMSK,
		Modulation_FSK,
		Modulation_8PSK,
		Modulation_16APSK,
		Modulation_32APSK,
		Modulation_LAST, // Keep me in last
	}PHY_MODULATION;
	static const char* strPHY_MODULATION[] = {
		"NULL",
		"DBPSK",
		"DQPSK",
		"CCK5.5",
		"CCK11",
		"BPSK",
		"QPSK",
		"8QAM",
		"16QAM",
		"32QAM",
		"64QAM",
		"128QAM",
		"256QAM",
		"GMSK",
		"FSK",
		"8PSK",
		"16APSK",
		"32APSK"
	};

	/* Enumeration for Transmitter status */
	typedef enum enum_tx_status
	{
		RX_OFF,
		RX_ON_IDLE,
		RX_ON_BUSY,
		TRX_ON_BUSY,
		SLEEP,
	}PHY_TX_STATUS;

	typedef struct stru_phytype_wireless
	{
		double dUpperFrequency;
		double dLowerFrequency;
		double dChannelBandwidth;
		PHY_TX_STATUS txStatus;
		double dTransmitPower_mw;
		double dTransmitPower_dbm;
		double* dReceivedPower_dbm;
		PHY_MODULATION modulation;
	}PHYPROP_WIRELESS,*PPHYPROP_WIRELESS;

	//Function Prototype
	_declspec(dllexport) PPHYPROP_WIRELESS get_phy_type_wireless(NETSIM_ID nDeviceId,NETSIM_ID nInterfaceId);
	_declspec(dllexport) void fn_NetSim_Wireless_SetFrequency(NETSIM_ID nDeviceId,
		NETSIM_ID nInterfaceId,
		double dUpperFrequency,
		double dLowerFrequency);
	_declspec(dllexport) void fn_NetSim_Wireless_SetChannelBandwidth(NETSIM_ID nDeviceId,
		NETSIM_ID nInterfaceId,
		double dChannelBandwidth);
	_declspec(dllexport) void fn_NetSim_Wireless_SetTxStatus(NETSIM_ID nDeviceId,
		NETSIM_ID nInterfaceId,
		PHY_TX_STATUS txStatus);
	_declspec(dllexport) void fn_NetSim_Wireless_SetTxPower_mw(NETSIM_ID nDeviceId,
		NETSIM_ID nInterfaceId,
		double txPower_mw);
	_declspec(dllexport) void fn_NetSim_Wireless_SetTxPower_dbm(NETSIM_ID nDeviceId,
		NETSIM_ID nInterfaceId,
		double txPower_dbm);
	_declspec(dllexport) void fn_NetSim_Wireless_SetRxPower_dbm(NETSIM_ID nDeviceId,
		NETSIM_ID nInterfaceId,
		NETSIM_ID nTransmitterId,
		double rxPower_dbm);
	_declspec(dllexport) void fn_NetSim_Wireless_SetRxPower_mw(NETSIM_ID nDeviceId,
		NETSIM_ID nInterfaceId,
		NETSIM_ID nTransmitterId,
		double rxPower_mw);
	_declspec(dllexport) void fn_NetSim_Wireless_GetRxPower(NETSIM_ID recvId,
		NETSIM_ID recvInterface,
		NETSIM_ID transmitterId,
		double* rxPower_mw,
		double* rxPower_dbm);
	_declspec(dllexport) void fn_NetSim_Wireless_SetModulation(NETSIM_ID nDeviceId,
		NETSIM_ID nInterfaceId,
		PHY_MODULATION modulation);
	_declspec(dllexport) void fn_NetSim_Wireless_SetProperty(NETSIM_ID nDeviceId,
		NETSIM_ID nInterfaceId,
		double dUpperFrequency,
		double dLowerFrequency,
		double dChannelBandwidth,
		double dTxPower_mw,
		PHY_MODULATION modulation,
		PHY_TX_STATUS txStatus);
	_declspec(dllexport) void fn_NetSim_Wireless_GetProperty(NETSIM_ID nDeviceId,
		NETSIM_ID nInterfaceId,
		double* dUpperFrequency,
		double* dLowerFrequency,
		double* dChannelBandwidth,
		double* dTxPower_mw,
		PHY_MODULATION* modulation,
		PHY_TX_STATUS* txStatus);



#ifdef  __cplusplus
}
#endif
#endif //_NETSIM_WIRELESS_H_
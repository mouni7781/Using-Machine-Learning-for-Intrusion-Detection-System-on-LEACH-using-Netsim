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
#include "../BatteryModel/BatteryModel.h"

/** 
	The Purpose of this function is to change the transceiver radio states  
*/
bool fn_NetSim_Zigbee_ChangeRadioState(NETSIM_ID nDeviceId, PHY_TX_STATUS nOldState, PHY_TX_STATUS nNewState)
{
	ptrIEEE802_15_4_PHY_VAR phy = WSN_PHY(nDeviceId);
	ptrBATTERY battery = phy->battery;
	bool isChange = true;
	if (battery)
	{
		isChange = battery_set_mode(battery, nNewState, ldEventTime);
	}

	if(isChange)
	{
		phy->nOldState = nOldState; 
		phy->nRadioState = nNewState; 
	}
	else
	{
		phy->nRadioState = RX_OFF;
		WSN_MAC(nDeviceId)->nNodeStatus = OFF;
	}
	return isChange;
}

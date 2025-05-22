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
#include "DSR.h"
#include "List.h"
/**
This function checks if the route cache of the device has the route to the Destination
*/
bool fn_NetSim_DSR_CheckRouteFound(NETSIM_IPAddress destAddress,DSR_DEVICE_VAR* devVar,NETSIM_IPAddress* nextHop,double dTime,DSR_ROUTE_CACHE** cache)
{
	*cache = DSR_FIND_CACHE(devVar,destAddress,dTime);
	if(*cache)
	{
		//Update the next hop
		*nextHop = (*cache)->address[1];
		return true;
	}	
	return false;
}



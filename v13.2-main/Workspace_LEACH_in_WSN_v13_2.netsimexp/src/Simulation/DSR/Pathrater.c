/************************************************************************************
* Copyright (C) 2015                                                               *
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
/****************************************************
This file contains code for Intrusion detection system in networks running DSR in Layer3.
This works only for UDP and not for TCP.
		 
The function add_to_blacklist(NETSIM_ID, NETSIM_IPAddress) adds the IP address of 
malicious node to black list indicating that it’s an intruder. 
		
The function find_ip_in_blacklist(NETSIM_ID,NETSIM_IPAddress) returns true if the device
is black listed.
		 
The function verify_route_reply(NETSIM_ID,DSR_RREP_OPTION*) verifies the IP address
obtained from route reply whether it is black listed or not. It returns true if the
route reply is not from malicious node.

The function blacklist_found(NETSIM_ID ,NETSIM_ID) deletes route entry from cache when
the node is black listed.

Code Flow - 
If The Node is a Malicious Node, then when a Route Reply is processed, the Function 
verifies the route reply in the route cache and checks for black listed node, i.e.,
malicious node.
When a malicious node is found, route entry is deleted from cache.
*****************************************************/

#include "main.h"
#include "../DSR/DSR.h"
#include "Pathrater.h"

PBLACKLIST* blacklist;
//Initializes the pathrater.
void pathrater_init()
{
	blacklist=(PBLACKLIST*)calloc(NETWORK->nDeviceCount+1,sizeof* blacklist);
}
//Closes the pathrater.
void pathrater_close()
{
	NETSIM_ID i;
	for(i=0;i<NETWORK->nDeviceCount+1;i++)
	{
		PBLACKLIST temp=blacklist[i];
		BLACKLIST_FREE_ALL(temp);
	}
	free(blacklist);
}
//Add malicious node to black list
void add_to_blacklist(NETSIM_ID dev_id,NETSIM_IPAddress ip)
{
	int n;
	PBLACKLIST black = BLACKLIST_ALLOC();
	black->ip=IP_COPY(ip);
	BLACKLIST_ADD(blacklist[dev_id],black);
	
}
//Check if the device is blacklisted
bool find_ip_in_blacklist(NETSIM_ID dev_id,NETSIM_IPAddress ip)
{
	int n; 
	PBLACKLIST black=blacklist[dev_id];
	while(black)
	{
		if(!IP_COMPARE(black->ip,ip))
		{
			return true;
		}
		BLACKLIST_NEXT(black);
	}
	return false;
}
//Verify route reply
bool verify_route_reply(NETSIM_ID dev_id,DSR_RREP_OPTION* rrep)
{
	int len = DSR_RREP_LEN(rrep);
	int i;
	bool ret = false;
	for(i=0;i<len;i++)
	{
		ret = find_ip_in_blacklist(dev_id,rrep->Address[i]);
		if(ret)
			return false;
	}
	return true;
}
//If node is blacklisted, delete route entry
_declspec(dllexport) void blacklist_found(NETSIM_ID dev_id,NETSIM_ID black_id)
{	
	int n = 0;
	NETSIM_IPAddress black_ip = dsr_get_dev_ip(black_id);
	NETSIM_IPAddress my_ip = dsr_get_dev_ip(dev_id);
	DSR_ROUTE_CACHE** cache = &(DSR_DEV_VAR(dev_id)->pstruRouteCache);
	//Add to blacklist
	add_to_blacklist(black_id,black_ip);
#ifdef _IDS_
	for (n = 0; n<(mal_dev_count); n++)
	{
		if (mal_devid[n] == black_id && mal_det_time[n] == 0)
			mal_det_time[n] = pstruEventDetails->dEventTime;//-mal_start_time[n];	
	}
#endif
	
	//Delete route from route cache
	DSR_DELETE_ENTRY_CACHE(cache,my_ip,black_ip);
}

/************************************************************************************
* Copyright (C) 2021																*
* TETCOS, Bangalore. India															*
*																					*
* Tetcos owns the intellectual property rights in the Product and its content.		*
* The copying, redistribution, reselling or publication of any or all of the		*
* Product or its content without express prior written consent of Tetcos is			*
* prohibited. Ownership and / or any other right relating to the software and all	*
* intellectual property rights therein shall remain at all times with Tetcos.		*
*																					*
* This source code is licensed per the NetSim license agreement.					*
*																					*
* No portion of this source code may be used as the basis for a derivative work,	*
* or used, for any purpose other than its intended use per the NetSim license		*
* agreement.																		*
*																					*
* This source code and the algorithms contained within it are confidential trade	*
* secrets of TETCOS and may not be used as the basis for any other software,		*
* hardware, product or service.														*
*																					*
* Author:    Shashi Kant Suman	                                                    *
*										                                            *
* ----------------------------------------------------------------------------------*/
#include "main.h"
#include "List.h"
#include "IP.h"

static bool isPublicIPReqd(NETSIM_ID d)
{
	NETSIM_ID i;
	bool ret = false;
	for (i = 0; i < DEVICE(d)->nNumOfInterface; i++)
	{
		if (DEVICE_INTERFACE(d, i + 1) &&
			DEVICE_INTERFACE(d, i + 1)->nInterfaceType == INTERFACE_WAN_ROUTER)
			return false;

		if (DEVICE_INTERFACE(d, i + 1) &&
			DEVICE_INTERFACE(d, i + 1)->szDefaultGateWay)
			ret = true;
	}
	return ret;
}

static NETSIM_IPAddress find_default_gateway(NETSIM_ID d,NETSIM_ID* di)
{
	NETSIM_ID i;
	for (i = 0; i < DEVICE(d)->nNumOfInterface; i++)
	{
		if (DEVICE_INTERFACE(d, i + 1) &&
			DEVICE_INTERFACE(d, i + 1)->szDefaultGateWay)
		{
			*di = i + 1;
			return DEVICE_INTERFACE(d, i + 1)->szDefaultGateWay;
		}
	}
	*di = 0;
	return NULL;
}

static NETSIM_IPAddress find_wan_ip(NETSIM_ID d, NETSIM_ID* wi)
{
	NETSIM_ID i;
	for (i = 0; i < DEVICE(d)->nNumOfInterface; i++)
	{
		if (DEVICE_INTERFACE(d, i + 1) &&
			DEVICE_INTERFACE(d, i + 1)->nInterfaceType == INTERFACE_WAN_ROUTER)
		{
			*wi = i + 1;
			return DEVICE_INTERFACE(d, i + 1)->szAddress;
		}
	}
	*wi = 0;
	return NULL;
}

static void assign_public_ip(NETSIM_ID d, NETSIM_IPAddress ip)
{
	NETSIM_ID i;
	for (i = 0; i < DEVICE(d)->nNumOfInterface; i++)
	{
		if (DEVICE_INTERFACE(d, i + 1) &&
			DEVICE_INTERFACE(d, i + 1)->szDefaultGateWay)
		{
			DEVICE_PUBLICIP(d, i + 1) = ip;
			fprintf(stdout, "Public IP of %d-%d is %s\n",
					d, i + 1, ip->str_ip);
		}
	}
}

typedef struct stru_ip_list
{
	NETSIM_IPAddress ip;
	NETSIM_IPAddress subnet;
	UINT prefix_len;
	struct stru_ip_list* next;
}IPLIST, * ptrIPLIST;

static void add_route(NETSIM_ID d, ptrIPLIST list,
					  NETSIM_IPAddress interfaceAddress,NETSIM_IPAddress nextHop)
{
	NETSIM_ID di = fn_NetSim_Stack_GetInterfaceIdFromIP(d, interfaceAddress);
	while (list)
	{
		NETSIM_IPAddress dest = IP_NETWORK_ADDRESS(list->ip, list->subnet,list->prefix_len);

		iptable_add(IP_WRAPPER_GET(d),
					dest, list->subnet, list->prefix_len,
					nextHop, 1, &interfaceAddress, &di, 1, "IP");
		list = list->next;
	}
}

static void setup_route_table_for_all_network(NETSIM_ID d, NETSIM_ID w, NETSIM_ID wi,
											  ptrIPLIST ipList)
{
	NETSIM_ID c = 0;
	NETSIM_ID ci = 0;
	NETSIM_ID di = 0;

	while (true)
	{
		NETSIM_IPAddress defaultGateway = find_default_gateway(d, &di);
		c = fn_NetSim_Stack_GetDeviceId_asIP(defaultGateway, &ci);
		add_route(c, ipList, defaultGateway, DEVICE_NWADDRESS(d, di));

		if (c == w)
			break;

		d = c;
	}
}

static void setup_route_table(NETSIM_ID d, NETSIM_ID w, NETSIM_ID wi)
{
	NETSIM_IPAddress pubIP = DEVICE_NWADDRESS(w, wi);

	ptrIPLIST head = NULL;
	ptrIPLIST tail = NULL;
	NETSIM_IPAddress defaultGateway = NULL;
	NETSIM_ID i;

	for (i = 0; i < DEVICE(d)->nNumOfInterface; i++)
	{
		if (DEVICE_INTERFACE(d, i + 1) &&
			DEVICE_NWADDRESS(d, i + 1))
		{
			if (DEVICE_INTERFACE(d, i + 1)->szDefaultGateWay && !defaultGateway)
			{
				defaultGateway = DEVICE_INTERFACE(d, i + 1)->szDefaultGateWay;
			}
			else
			{
				ptrIPLIST l = calloc(1, sizeof * l);
				l->ip = DEVICE_INTERFACE(d, i + 1)->szAddress;
				l->subnet = DEVICE_INTERFACE(d, i + 1)->szSubnetMask;
				l->prefix_len = DEVICE_INTERFACE(d, i + 1)->prefix_len;;

				if (head)
				{
					tail->next = l;
					tail = l;
				}
				else
				{
					head = l;
					tail = l;
				}
			}
		}
	}

	if(head)
		setup_route_table_for_all_network(d, w, wi, head);

	while (head)
	{
		ptrIPLIST t = head;
		head = head->next;
		free(t);
	}
}

void set_public_ip(NETSIM_ID d)
{
	if (!isPublicIPReqd(d))
		return;

	NETSIM_ID c = d;
	NETSIM_ID ci = 0;
	while (true)
	{
		NETSIM_IPAddress defaultGateway = find_default_gateway(c, &ci);
		if (!defaultGateway)
			break;

		c = fn_NetSim_Stack_GetDeviceId_asIP(defaultGateway, &ci);

		NETSIM_IPAddress wanIP = find_wan_ip(c, &ci);

		if (wanIP)
		{
			assign_public_ip(d, wanIP);
			c = fn_NetSim_Stack_GetDeviceId_asIP(wanIP, &ci);
			setup_route_table(d, c, ci);
			break;
		}
	}
	
}

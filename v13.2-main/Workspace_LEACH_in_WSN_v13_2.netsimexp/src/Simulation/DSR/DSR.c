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
#include "Pathrater.h"
int fn_NetSim_DSR_Init_F(struct stru_NetSim_Network *NETWORK_Formal,
						 NetSim_EVENTDETAILS *pstruEventDetails_Formal,
						 char *pszAppPath_Formal,
						 char *pszWritePath_Formal,
						 int nVersion_Type,
						 void **fnPointer);
int fn_NetSim_DSR_Configure_F(void** var);
int fn_NetSim_DSR_CopyPacket_F(const NetSim_PACKET* destPacket,const NetSim_PACKET* srcPacket);
int fn_NetSim_DSR_FreePacket_F(NetSim_PACKET* packet);
int fn_NetSim_DSR_Metrics_F(PMETRICSWRITER filename);



/**
DSR Init function initializes the DSR parameters.
*/
_declspec(dllexport) int fn_NetSim_DSR_Init(struct stru_NetSim_Network *NETWORK_Formal,
											NetSim_EVENTDETAILS *pstruEventDetails_Formal,
											char *pszAppPath_Formal,
											char *pszWritePath_Formal,
											int nVersion_Type,
											void **fnPointer)
{
	fn_NetSim_DSR_Init_F(NETWORK_Formal,pstruEventDetails_Formal,pszAppPath_Formal,pszWritePath_Formal,nVersion_Type,fnPointer);
#ifdef _NETSIM_PATHRATER_
	pathrater_init();
#endif // _NETSIM_PATHRATER_
	return 1;
}
/**
This is the DSR configure function.
*/
_declspec(dllexport) int fn_NetSim_DSR_Configure(void** var)
{
	return fn_NetSim_DSR_Configure_F(var);
}
/**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is the DSR Run function which gets called by the IP layer for routing the data
by the DSR Network Routing Protocol. It includes the events NETWORK_OUT_EVENT, 
NETWORK_IN_EVENT and TIMER_EVENT.

NETWORK_OUT_EVENT - 
	It process the Data Packets which arrive at the NetworkOut layer to route the packet.
NETWORK_IN_EVENT - 
	It processes Data Packets, Route Request Packets, Route Reply Packets, Route Error 
	Packets, Ack Packet
TIMER_EVENT -
	It Process the DSR Route Request timeout and the DSR Maintenance Buffer timeout.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*/
_declspec(dllexport) int fn_NetSim_DSR_Run()
{
	set_dsr_curr();
	if (!isDsrConfigured(pstruEventDetails->nDeviceId, dsr_get_curr_if()))
		return -1;
	switch(pstruEventDetails->nEventType)
	{
	case NETWORK_OUT_EVENT:
		switch(pstruEventDetails->nSubEventType)
		{
		case 0: //Packet from upper layer or Network in
		default:
			{
			if (!isDSRPACKET(pstruEventDetails->pPacket))
			{
#ifdef _SINKHOLE_
				if (fn_NetSim_DSR_MaliciousNode(pstruEventDetails->nDeviceId, pstruEventDetails->dEventTime))
				{
					fn_NetSim_Packet_FreePacket(pstruEventDetails->pPacket);
					break;
				}
				else
#endif // _SINKHOLE_
				{
					NETSIM_IPAddress n1, n2;
					NETSIM_IPAddress dest = pstruEventDetails->pPacket->pstruNetworkData->szDestIP;
					NETSIM_IPAddress ip = dsr_get_curr_ip();
					if (ip->type != dest->type)
						return 0;
					n1 = IP_NETWORK_ADDRESS(ip,
						DEVICE_INTERFACE(pstruEventDetails->nDeviceId, 1)->szSubnetMask,
						DEVICE_INTERFACE(pstruEventDetails->nDeviceId, 1)->prefix_len);
					n2 = IP_NETWORK_ADDRESS(dest,
						DEVICE_INTERFACE(pstruEventDetails->nDeviceId, 1)->szSubnetMask,
						DEVICE_INTERFACE(pstruEventDetails->nDeviceId, 1)->prefix_len);

					if (!IP_COMPARE(n1, n2))
						DSR_PACKET_PROCESSING();
					else
						return 0; //Other network packet
				}
			}
				else
				{
					//DSR control packet
					//No processing just forward
				}
			}
			break;
		}
		if(pstruEventDetails->pPacket)
			pstruEventDetails->nInterfaceId = dsr_get_curr_if();
		else
			pstruEventDetails->nInterfaceId = 0;
		break;
	case NETWORK_IN_EVENT:
		{
			switch(pstruEventDetails->nSubEventType)
			{
			case 0:
			default:
				{
					switch(pstruEventDetails->pPacket->nControlDataType)
					{
					default://Data packet or other layer control packet
#ifdef _SINKHOLE_
						if (fn_NetSim_DSR_MaliciousNode(pstruEventDetails->nDeviceId, pstruEventDetails->dEventTime))
						{
							fn_NetSim_DSR_MaliciousProcessSourceRouteOption(pstruEventDetails);
							return 1;
						}
#endif // _SINKHOLE_

						if(pstruEventDetails->pPacket->pstruNetworkData->Packet_RoutingProtocol)
						{
							DSR_PROCESS_SRC_ROUTE();
							if(pstruEventDetails->pPacket && pstruEventDetails->pPacket->pstruNetworkData)
								pstruEventDetails->pPacket->pstruNetworkData->nTTL--;
							//pstruEventDetails->pPacket=NULL;
						}
						break;
					case ctrlPacket_ROUTE_REQUEST:
						DSR_PROCESS_RREQ();
						pstruEventDetails->pPacket=NULL;
						break;
					case ctrlPacket_ROUTE_REPLY:
						DSR_PROCESS_RREP();
						pstruEventDetails->pPacket=NULL;
						break;
					case ctrlPacket_ROUTE_ERROR:
						DSR_PROCESS_RERR();
						pstruEventDetails->pPacket=NULL;

						break;
					case ctrlPacket_ACK:
						DSR_PROCESS_ACK();
						pstruEventDetails->pPacket=NULL;

					}
				}
				break;
			case subevent_PROCESS_RERR:
				DSR_PROCESS_RERR();
				pstruEventDetails->pPacket=NULL;
			break;
			}
		}
		break;
	case TIMER_EVENT:
		{
			switch(pstruEventDetails->nSubEventType)
			{
				case subevent_RREQ_TIMEOUT:
					DSR_RREQ_TIMEOUT();
					
				break;
				case subevent_MAINT_TIMEOUT:
					DSR_MAINT_TIMEOUT();
				break;
			}
		}
		break;
	}
	
	return 1;
}
/**
	This function is called by NetworkStack.dll, while writing the evnt trace 
	to get the sub event as a string.
*/
_declspec(dllexport) char* fn_NetSim_DSR_Trace(NETSIM_ID nSubeventid)
{
	switch(nSubeventid)
	{
	case subevent_RREQ_TIMEOUT:
		return "DSR_RREQ_TIMEOUT";
	case subevent_MAINT_TIMEOUT:
		return "DSR_MAINT_TIMEOUT";
	case subevent_PROCESS_RERR:
		return "DSR_PROCESS_RERR";
	default:
		return "DSR_UNKNOWN";
	}
}
/**
	This function is called by NetworkStack.dll, to copy the DSR protocol
	from source packet to destination.
*/
_declspec(dllexport) int fn_NetSim_DSR_CopyPacket(const NetSim_PACKET* destPacket,const NetSim_PACKET* srcPacket)
{
	return fn_NetSim_DSR_CopyPacket_F(destPacket,srcPacket);
}
/**
	This function is called by NetworkStack.dll, to free the TCP protocol.
*/
_declspec(dllexport) int fn_NetSim_DSR_FreePacket(NetSim_PACKET* packet)
{
	return fn_NetSim_DSR_FreePacket_F(packet);
}
/**
This function call WLAN Metrics function in lib file to write the Metrics in Metrics.txt.	
*/
_declspec(dllexport) int fn_NetSim_DSR_Metrics(PMETRICSWRITER filename)
{
#ifdef _IDS_
	PMETRICSNODE table = init_metrics_node(MetricsNode_Table,
		"Custom_IDS_Metrics", NULL);

	add_table_heading(table, "DeviceID", true, 0);
	add_table_heading(table, "Start Time (micro sec)", true, 0);
	add_table_heading(table, "Detection Time (micro sec)", true, 0);

	int i;
	for (i = 0; i < mal_dev_count; i++)
	{
		add_table_row_formatted(false, table, "%d,%f,%f,",
			mal_devid[i], mal_start_time[i],
			mal_det_time[i]);
	}
	PMETRICSNODE menu = init_metrics_node(MetricsNode_Menu,
		"CUSTOM_METRICS", NULL);
	add_node_to_menu(menu, table);
	write_metrics_node(filename, WriterPosition_Current, NULL, menu);
	delete_metrics_node(menu);
#endif // _IDS_

	return fn_NetSim_DSR_Metrics_F(filename);
}
/**
	This function is called by NetworkStack.dll, once simulation end to free the 
	allocated memory for the network.	
*/
_declspec(dllexport) int fn_NetSim_DSR_Finish()
{
#ifdef _NETSIM_PATHRATER_
	pathrater_close();
#endif // _NETSIM_PATHRATER_

	return fn_NetSim_DSR_Finish_F();
}
/**
This function will return the string to write packet trace heading.
*/
_declspec(dllexport) char* fn_NetSim_DSR_ConfigPacketTrace()
{
	return "";
}
/**
 This function will return the string to write packet trace.																									
*/
_declspec(dllexport) char* fn_NetSim_DSR_WritePacketTrace()
{
	return "";
}

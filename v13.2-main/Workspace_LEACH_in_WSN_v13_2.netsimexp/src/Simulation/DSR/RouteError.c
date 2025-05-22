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
int fn_NetSim_DSR_ForwardRERR(NetSim_EVENTDETAILS* pstruEventDetails);
/**
This function gets the previous hop to send the route error to.
*/
NETSIM_IPAddress fnGetPrevHop(NetSim_PACKET* packet)
{
	DSR_OPTION_HEADER* option = packet->pstruNetworkData->Packet_RoutingProtocol;
	DSR_SOURCE_ROUTE_OPTION* srcRouteOption = option->options;
	unsigned int nlength = (srcRouteOption->nOptDataLen-2)/4;
	if(!srcRouteOption->Address || nlength == 1 || srcRouteOption->nSegsLeft == nlength)
		return packet->pstruNetworkData->szSourceIP;
	return srcRouteOption->Address[srcRouteOption->nSegsLeft];
}
/**
This function generates a route error packet.
*/
int fn_NetSim_DSR_GenerateRERR(DSR_MAINT_BUFFER* maintBuffer,NetSim_EVENTDETAILS* pstruEventDetails)
{
	unsigned int length;
	unsigned int loop;
	NETSIM_ID recv;
	NetSim_EVENTDETAILS pevent;
	NetSim_PACKET* packet = maintBuffer->packetList;
	DSR_OPTION_HEADER* option = (DSR_OPTION_HEADER*)packet->pstruNetworkData->Packet_RoutingProtocol;
	DSR_SOURCE_ROUTE_OPTION* srcRouteOption = (DSR_SOURCE_ROUTE_OPTION*)option->options;
	NETSIM_IPAddress nexthop = fnGetPrevHop(packet);
	NetSim_PACKET* rerrPacket = fn_NetSim_DSR_GenerateCtrlPacket(pstruEventDetails->nDeviceId,
		maintBuffer->source,
		maintBuffer->source,
		pstruEventDetails->dEventTime,
		ctrlPacket_ROUTE_ERROR);
	DSR_RERR_OPTION* rerrOption=calloc(1,sizeof* rerrOption);
	option = calloc(1,sizeof* option);
	rerrPacket->pstruNetworkData->Packet_RoutingProtocol = option;
	option->nNextHeader = NO_NEXT_HEADER;
	option->optType = optType_RouteError;
	option->options = rerrOption;
	rerrOption->nErrorType = NODE_UNREACHABLE;
	rerrOption->errorSourceAddress = IP_COPY(rerrPacket->pstruNetworkData->szSourceIP);
	rerrOption->errorDestinationAddress = IP_COPY(rerrPacket->pstruNetworkData->szDestIP);
	rerrOption->TypeSpecificInformation = IP_COPY(fn_NetSim_Stack_GetIPAddressAsId(maintBuffer->nextHop,1));
	rerrOption->nOptDataLen = DSR_RERR_SIZE_FIXED;
	rerrOption->nOptionType = optType_RouteError;
	length = (srcRouteOption->nOptDataLen-2)/4-srcRouteOption->nSegsLeft;
	rerrOption->length = length;
	rerrOption->nSegsLeft = length;
	if (length)
	{
		rerrOption->Address = calloc(length, sizeof* rerrOption->Address);
		for (loop = 0; loop < length; loop++)
			rerrOption->Address[loop] = IP_COPY(srcRouteOption->Address[length - loop - 1]);
	}
		
	option->nPayloadLength = DSR_RERR_SIZE_FIXED;
	rerrPacket->pstruNetworkData->nTTL= length +1;
	rerrPacket->pstruNetworkData->dOverhead = DSR_RERR_SIZE_FIXED+DSR_OPTION_HEADER_SIZE;
	rerrPacket->pstruNetworkData->dPacketSize =  rerrPacket->pstruNetworkData->dOverhead;
	rerrPacket->pstruNetworkData->szNextHopIp = IP_COPY(nexthop);
	rerrPacket->nReceiverId = fn_NetSim_Stack_GetDeviceId_asIP(nexthop,&recv);
	
	//Add network out event to transmit packet
	memcpy(&pevent,pstruEventDetails,sizeof* pstruEventDetails);
	pevent.nApplicationId = 0;
	pevent.nEventType = NETWORK_IN_EVENT;
	pevent.nPacketId = 0;
	pevent.nProtocolId = NW_PROTOCOL_DSR;
	pevent.nSegmentId = 0;
	pevent.nSubEventType = subevent_PROCESS_RERR;
	pevent.pPacket = rerrPacket;
	pevent.dPacketSize = rerrPacket->pstruNetworkData->dPacketSize;
	pevent.szOtherDetails = NULL;
	fnpAddEvent(&pevent);
	//Update the metrics
	DSR_DEV_VAR(pstruEventDetails->nDeviceId)->dsrMetrics.rerrSent++;
	return 1;
}
/**
This function process the route error that is received. It deletes the entry of the route to 
the target from the route cache and then forwards the Route error to the previous HOP.
*/
int fn_NetSim_DSR_ProcessRerr(NetSim_EVENTDETAILS* pstruEventDetails)
{
	NetSim_PACKET* packet = pstruEventDetails->pPacket;
	DSR_OPTION_HEADER* option = (DSR_OPTION_HEADER*)packet->pstruNetworkData->Packet_RoutingProtocol;
	DSR_RERR_OPTION* rerr = (DSR_RERR_OPTION*)option->options;
	
	//Delete entry from route cache
	DSR_DELETE_ENTRY_CACHE(&(DSR_DEV_VAR(pstruEventDetails->nDeviceId)->pstruRouteCache),
		rerr->errorSourceAddress,
		(NETSIM_IPAddress)rerr->TypeSpecificInformation);
	fn_NetSim_DSR_ForwardRERR(pstruEventDetails);
	return 1;
}
/**
This function frees the route error packet if RERR packet reaches the final destination.
Otherwise, it forwards the REER packet to the previous HOP.
*/
int fn_NetSim_DSR_ForwardRERR(NetSim_EVENTDETAILS* pstruEventDetails)
{
	NETSIM_ID interfaceId;
	NetSim_PACKET* packet = pstruEventDetails->pPacket;
	DSR_OPTION_HEADER* option = packet->pstruNetworkData->Packet_RoutingProtocol;
	DSR_RERR_OPTION* rerr = option->options;
	packet->pstruNetworkData->szSourceIP = dsr_get_curr_ip();
	if(rerr->nSegsLeft == 0)
	{
		fn_NetSim_Packet_FreePacket(pstruEventDetails->pPacket);
		pstruEventDetails->pPacket = NULL;
		return 0;
	}
	rerr->nSegsLeft--;
	
	if(rerr->nSegsLeft)
		packet->pstruNetworkData->szNextHopIp = IP_COPY(rerr->Address[rerr->nSegsLeft]);
	else
		packet->pstruNetworkData->szNextHopIp = IP_COPY(packet->pstruNetworkData->szDestIP);
	packet->nTransmitterId = pstruEventDetails->nDeviceId;
	packet->nReceiverId = fn_NetSim_Stack_GetDeviceId_asIP(packet->pstruNetworkData->szNextHopIp,&interfaceId);
	//Add the network out event
	pstruEventDetails->nEventType = NETWORK_OUT_EVENT;
	pstruEventDetails->nSubEventType = 0;
	fnpAddEvent(pstruEventDetails);
	//update the metrics
	DSR_DEV_VAR(pstruEventDetails->nDeviceId)->dsrMetrics.rerrForwarded++;
	return 1;
}





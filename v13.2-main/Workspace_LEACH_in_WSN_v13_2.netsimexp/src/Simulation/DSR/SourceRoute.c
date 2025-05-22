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
#include "List.h"
#include "DSR.h"
/**
This function adds the source route option
*/
int fn_NetSim_DSR_AddSourceRouteOption(NetSim_PACKET* packet,DSR_ROUTE_CACHE* cache)
{
//	unsigned int loop;
	unsigned int length;
	DSR_OPTION_HEADER* option;
	DSR_SOURCE_ROUTE_OPTION* srcRouteOption;
	option = (DSR_OPTION_HEADER*)packet->pstruNetworkData->Packet_RoutingProtocol;
	if(option && option->optType == optType_SourceRoute)
	{
		//Already added and processed
		return 1;
	}
	option = calloc(1,sizeof* option);
	srcRouteOption=calloc(1,sizeof* srcRouteOption);
	option->options = srcRouteOption;
	option->nNextHeader = NO_NEXT_HEADER;
	option->optType = optType_SourceRoute;
	srcRouteOption->F = cache->F;
	srcRouteOption->L = cache->L;
	srcRouteOption->nSalvage = 0;
	srcRouteOption->nOptionType = optType_SourceRoute;
	srcRouteOption->nReserved = 0;
	length = fn_NetSim_DSR_FillAddress(srcRouteOption,cache,dsr_get_curr_ip(),packet->pstruNetworkData->szDestIP);
	srcRouteOption->nSegsLeft = length;
	srcRouteOption->nOptDataLen = length*4+2;
	packet->pstruNetworkData->Packet_RoutingProtocol = option;
	packet->pstruNetworkData->nRoutingProtocol = NW_PROTOCOL_DSR;
	option->nPayloadLength = DSR_SOURCEROUTE_SIZE_FIXED+length*4;
	packet->pstruNetworkData->dOverhead = DSR_OPTION_HEADER_SIZE+option->nPayloadLength;
	packet->pstruNetworkData->dPayload = packet->pstruTransportData->dPacketSize;
	packet->pstruNetworkData->dPacketSize = packet->pstruNetworkData->dOverhead +
		packet->pstruNetworkData->dPayload;
	return 1;
}
/**
This function process the data packet. it the device is the intended target, 
it adds a Transport_In_Event. Else it adds Network_Out_Event
*/
int fn_NetSim_DSR_ProcessSourceRouteOption(NetSim_EVENTDETAILS* pstruEventDetails)
{
	NetSim_PACKET* packet = pstruEventDetails->pPacket;
	DSR_OPTION_HEADER* option;
	DSR_SOURCE_ROUTE_OPTION* srcRouteOption;
	//LEACH
	if (fn_NetSim_LEACH_CheckDestination(pstruEventDetails->nDeviceId, get_first_dest_from_packet(pstruEventDetails->pPacket)))
	{
		pstruEventDetails->nEventType = TRANSPORT_IN_EVENT;
		fnpAddEvent(pstruEventDetails);
		return 1;
	}
	else
	{
		pstruEventDetails->pPacket->pstruNetworkData->szNextHopIp = NULL;
		pstruEventDetails->pPacket->nReceiverId = 0;
		pstruEventDetails->nEventType = NETWORK_OUT_EVENT;
		fnpAddEvent(pstruEventDetails);
		return 1;
	}
	//LEACH
	option = (DSR_OPTION_HEADER*)packet->pstruNetworkData->Packet_RoutingProtocol;
	if(!option)
		return 0;
	if(option->ackRequestOption)
		DSR_PROCESS_ACK_REQUEST(packet);
	if(option && option->optType == optType_SourceRoute)
	{
		//update the metrics
		DSR_DEV_VAR(pstruEventDetails->nDeviceId)->dsrMetrics.packetReceived++;
		srcRouteOption = option->options;
		//LEACH
		if (!IP_COMPARE(packet->pstruNetworkData->szDestIP, dsr_get_curr_ip()))
		{
			//Add Transport in event
			pstruEventDetails->nEventType = TRANSPORT_IN_EVENT;
			fnpAddEvent(pstruEventDetails);
			return 1;
		}
		if (srcRouteOption->nSegsLeft)
			srcRouteOption->nSegsLeft -= 1;
		//Add network out event
		pstruEventDetails->pPacket->pstruNetworkData->szNextHopIp = NULL;
		pstruEventDetails->nEventType = NETWORK_OUT_EVENT;
		fnpAddEvent(pstruEventDetails);
		//LEACH
		return 1;

	}
	return 0;
}



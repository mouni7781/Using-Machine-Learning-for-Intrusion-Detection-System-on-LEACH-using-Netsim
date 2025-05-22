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
* Author:    Shashi Kant Suman                                                     *
*                                                                                  *
* ---------------------------------------------------------------------------------*/
#include "main.h"
#include "DSR.h"
#include "List.h"
#include "Pathrater.h"
/**
This function Generates a route reply to be sent to the source.
*/
int fn_NetSim_DSR_GenerateRREP(NetSim_PACKET* rreqPacket,NetSim_EVENTDETAILS* pstruEventDetails)
{
	NETSIM_ID nLoop;
	unsigned int addressCount;
	DSR_RREQ_OPTION* rreq = ((DSR_OPTION_HEADER*)rreqPacket->pstruNetworkData->Packet_RoutingProtocol)->options;
	NetSim_PACKET* packet = fn_NetSim_DSR_GenerateCtrlPacket(
		pstruEventDetails->nDeviceId,
		rreqPacket->nSourceId,
		rreqPacket->nSourceId,
		pstruEventDetails->dEventTime,
		ctrlPacket_ROUTE_REPLY);
	DSR_OPTION_HEADER* option = calloc(1,sizeof* option);
	DSR_RREP_OPTION* rrep = calloc(1,sizeof* rrep);

	option->nFlowState = 0;
	option->nNextHeader = NO_NEXT_HEADER;
	option->nPayloadLength = DSR_RREP_SIZE_FIXED;
	option->nReserved = 0;
	option->optType = optType_RouteReply;
	packet->pstruNetworkData->Packet_RoutingProtocol = option;

	option->options = rrep;
	rrep->nOptionType = optType_RouteReply;
	rrep->nLastHopExternal = 0;
	rrep->nReserved = 0;
	addressCount = DSR_RREQ_LEN(rreq);
	addressCount+=2; //Add for initiator and target
	rrep->nOptDataLen = (addressCount-1)*4+3;
	rrep->Address = calloc(addressCount,sizeof* rrep->Address);
	rrep->Address[0]= dsr_get_dev_ip(rreqPacket->nSourceId);
	rrep->Address[addressCount - 1] = dsr_get_curr_ip();


	for(nLoop=1;nLoop<addressCount-1;nLoop++)
		rrep->Address[nLoop] = IP_COPY(rreq->address[nLoop-1]);
	packet->nReceiverId = fn_NetSim_Stack_GetDeviceId_asIP(rrep->Address[addressCount-2],&(nLoop));
	IP_FREE(packet->pstruNetworkData->szNextHopIp);
	packet->pstruNetworkData->szNextHopIp = IP_COPY(rrep->Address[addressCount-2]);
	packet->pstruNetworkData->dPacketSize = rrep->nOptDataLen+DSR_OPTION_HEADER_SIZE;
	packet->pstruNetworkData->dOverhead =  rrep->nOptDataLen+DSR_OPTION_HEADER_SIZE;
	packet->pstruNetworkData->nTTL = addressCount+2;

	//Generate Network out event
	pstruEventDetails->dEventTime+=fn_NetSim_DSR_GetBroadCastJitter();
	pstruEventDetails->dPacketSize = rrep->nOptDataLen+DSR_OPTION_HEADER_SIZE;
	pstruEventDetails->nEventType = NETWORK_OUT_EVENT;
	pstruEventDetails->nProtocolId = NW_PROTOCOL_DSR;
	pstruEventDetails->nSubEventType = 0;
	pstruEventDetails->pPacket = packet;
	fnpAddEvent(pstruEventDetails);
	//Update the metrics
	DSR_DEV_VAR(pstruEventDetails->nDeviceId)->dsrMetrics.rrepSent++;
	return 0;
}
/**
This function processes the route reply.
It Updates the route cache of the device and adds the new route to the destination in the 
route cache.
It checks the Send Buffer if any packets are in the list to be sent.
Then it forwards the RREP packet to the prev HOP.
*/
int fn_NetSim_DSR_ProcessRREP(NetSim_EVENTDETAILS* pstruEventDetails)
{
	DSR_OPTION_HEADER* option = (DSR_OPTION_HEADER*)pstruEventDetails->pPacket->pstruNetworkData->Packet_RoutingProtocol;
	DSR_RREP_OPTION* rrep = (DSR_RREP_OPTION*)option->options;
#ifdef _NETSIM_PATHRATER_
	if (!verify_route_reply(pstruEventDetails->nDeviceId, rrep))
		return -1; //route reply contains blacklist ip
#endif // _NETSIM_PATHRATER_

	DSR_UPDATE_ROUTE_CACHE(DSR_RREP_LEN(rrep),
		rrep->Address,
		pstruEventDetails->dEventTime);
	DSR_CHECK_SEND_BUFFER(pstruEventDetails->nDeviceId,
		pstruEventDetails->dEventTime);
	fn_NetSim_DSR_ForwardRREP();
	return 0;
}

static void determine_position(DSR_ROUTE_CACHE* cache, int* my, int* de, NETSIM_IPAddress myIP, NETSIM_IPAddress dest)
{
	UINT i;
	for (i = 0; i < cache->nLength; i++)
	{
		if (!IP_COMPARE(cache->address[i], myIP))
			*my = i + 1;
		if (!IP_COMPARE(cache->address[i], dest))
		{
			*de = i + 1;
			return;
		}
	}
}

/**
This functions generates a route reply if the route to destination is there in the
route cache.
*/
bool fn_NetSim_DSR_GenerateRREPUsingRouteCache(DSR_DEVICE_VAR* devVar,
	NetSim_PACKET* rreqPacket,
	double dTime,
	NetSim_EVENTDETAILS* pstruEventDetails)
{
	DSR_RREQ_OPTION* rreq = ((DSR_OPTION_HEADER*)(rreqPacket->pstruNetworkData->Packet_RoutingProtocol))->options;
	DSR_ROUTE_CACHE* cache = DSR_FIND_CACHE(devVar,rreq->targetAddress,dTime);
	if(!DSR_VALIDATE_CACHE(cache,rreq->address,DSR_RREQ_LEN(rreq)))
	{
		return false;
	}
	else
	{
		int nLoop;
		unsigned int addressCount;
		int my=-1, de=-1;
		NETSIM_ID n=0;
		NetSim_PACKET* packet = fn_NetSim_DSR_GenerateCtrlPacket(
			pstruEventDetails->nDeviceId,
			rreqPacket->nSourceId,
			rreqPacket->nSourceId,
			pstruEventDetails->dEventTime,
			ctrlPacket_ROUTE_REPLY);
		DSR_OPTION_HEADER* option = calloc(1,sizeof* option);
		DSR_RREP_OPTION* rrep = calloc(1,sizeof* rrep);

		determine_position(cache, &my, &de,
						   dsr_get_dev_ip(rreqPacket->nSourceId),
						   rreq->targetAddress);

		UINT len = my == -1 ? de : de - my;

		option->nFlowState = 0;
		option->nNextHeader = NO_NEXT_HEADER;
		option->nPayloadLength = DSR_RREP_SIZE_FIXED;
		option->nReserved = 0;
		option->optType = optType_RouteReply;
		packet->pstruNetworkData->Packet_RoutingProtocol = option;

		option->options = rrep;
		rrep->nOptionType = optType_RouteReply;
		rrep->nLastHopExternal = 0;
		rrep->nReserved = 0;
		addressCount = DSR_RREQ_LEN(rreq) + len;//cache->nLength;
		addressCount+=1; //Add for initiator
		
		rrep->nOptDataLen = (addressCount-1)*4+3;
		rrep->Address = calloc(addressCount,sizeof* rrep->Address);
		
		rrep->Address[0]= dsr_get_dev_ip(rreqPacket->nSourceId);
		
		for(nLoop=1;nLoop<=DSR_RREQ_LEN(rreq);nLoop++)
			rrep->Address[nLoop] = rreq->address[nLoop-1];

		packet->nReceiverId = fn_NetSim_Stack_GetDeviceId_asIP(rrep->Address[nLoop-1],&n);
		packet->pstruNetworkData->szNextHopIp = IP_COPY(rrep->Address[nLoop-1]);
		packet->pstruNetworkData->nTTL = 1;
		n = my == -1 ? 0 : my;
		for(;(unsigned int)nLoop<addressCount;nLoop++)
			rrep->Address[nLoop] = IP_COPY(cache->address[n++]);


		//Generate Network out event
		pstruEventDetails->dEventTime+=fn_NetSim_DSR_GetBroadCastJitter();
		pstruEventDetails->dPacketSize = rrep->nOptDataLen+DSR_OPTION_HEADER_SIZE;
		pstruEventDetails->nEventType = NETWORK_OUT_EVENT;
		pstruEventDetails->nProtocolId = NW_PROTOCOL_DSR;
		pstruEventDetails->nSubEventType = 0;
		pstruEventDetails->pPacket = packet;
		fnpAddEvent(pstruEventDetails);
		//Update the metrics
		devVar->dsrMetrics.rrepSent++;
		return true;
	}
}
/**
This function checks if the current device is the last HOP. If it is, it drops the packet and
no further forwarding is done.

Else, the RREP is forwarded to the Previous HOP
*/
int fn_NetSim_DSR_ForwardRREP()
{
	NetSim_PACKET* packet = pstruEventDetails->pPacket;
	NETSIM_ID nDeviceId = pstruEventDetails->nDeviceId;

	DSR_OPTION_HEADER* option = (DSR_OPTION_HEADER*)packet->pstruNetworkData->Packet_RoutingProtocol;
	DSR_RREP_OPTION* rrep = (DSR_RREP_OPTION*)option->options;
	int length = DSR_RREP_LEN(rrep);
	NETSIM_ID n=0;
	NETSIM_IPAddress nextHop=NULL;

	if(!IP_COMPARE(rrep->Address[0],dsr_get_curr_ip()))
	{
		DSR_RREQ_TABLE* table = getRREQTable(rrep->Address[DSR_RREP_LEN(rrep)-1],
			DSR_DEV_VAR(pstruEventDetails->nDeviceId)->pstruRREQTable);
		if(table)
		{
			table->flag = true;//RREP received
			fnDeleteEvent(table->nEventId);
		}
		fn_NetSim_Packet_FreePacket(packet);
		pstruEventDetails->pPacket=NULL;
		return 1; //Last hop no need to forward
	}
	while(n<(NETSIM_ID)length)
	{

		if(!IP_COMPARE(rrep->Address[n],dsr_get_curr_ip()))
		{
			nextHop = rrep->Address[n-1];
			break;
		}
		n++;
	}
	if(nextHop)
	{
		IP_FREE(packet->pstruNetworkData->szNextHopIp);
		packet->pstruNetworkData->szNextHopIp = IP_COPY(nextHop);
		packet->pstruNetworkData->szGatewayIP = dsr_get_curr_ip();
		packet->nTransmitterId = nDeviceId;
		packet->nReceiverId = fn_NetSim_Stack_GetDeviceId_asIP(nextHop,&n);
		//Add Network out event
		pstruEventDetails->nEventType = NETWORK_OUT_EVENT;
		fnpAddEvent(pstruEventDetails);
		//Update the metrics
		DSR_DEV_VAR(nDeviceId)->dsrMetrics.rrepForwarded++;
	}
	return 0;
}


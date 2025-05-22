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
#include "List.h"
#include "DSR.h"
int fn_NetSim_DSR_DeleteRREQTable(NETSIM_IPAddress targetAddress,NETSIM_ID nDeviceId);
double getRREQBackoff(DSR_RREQ_TABLE* table);
bool fn_NetSim_DSR_CheckIPinIPList(NETSIM_IPAddress* ipList,int length,NETSIM_IPAddress ip);
bool fn_NetSim_DSR_CheckEntryInRREQTable(DSR_RREQ_OPTION* rreq,DSR_RREQ_TABLE* table);
/**
This function initiates a route request to the target.
*/
NetSim_PACKET* fn_NetSim_DSR_InitRouteRequest(NETSIM_IPAddress target,DSR_RREQ_TABLE** rreqTable,
	NETSIM_ID nDeviceId,
	double dTime,
	NetSim_EVENTDETAILS* pstruEventDetails)
{
	NetSim_EVENTDETAILS pevent;
	NetSim_PACKET* rreq = fn_NetSim_DSR_GenerateCtrlPacket(nDeviceId,
		0,0,/*BROADCAST*/
		dTime,
		ctrlPacket_ROUTE_REQUEST);
	DSR_OPTION_HEADER* option=(DSR_OPTION_HEADER*)calloc(1,sizeof* option);
	DSR_RREQ_OPTION* rreqOption=(DSR_RREQ_OPTION*)calloc(1,sizeof* rreqOption);
	DSR_RREQ_TABLE* table;
	rreq->pstruNetworkData->dPacketSize=DSR_RREQ_SIZE_FIXED+DSR_OPTION_HEADER_SIZE;
	rreq->pstruNetworkData->dOverhead=DSR_RREQ_SIZE_FIXED+DSR_OPTION_HEADER_SIZE;
	rreq->pstruNetworkData->dPayload=0;

	rreq->pstruNetworkData->Packet_RoutingProtocol = option;
	option->nFlowState=0;
	option->nNextHeader=NO_NEXT_HEADER;
	option->nReserved = 0;
	option->nPayloadLength=DSR_RREQ_SIZE_FIXED;
	option->optType = optType_RouteRequest;

	option->options = rreqOption;
	rreqOption->nOptionType = optType_RouteRequest;
	rreqOption->nOptDataLen = DSR_RREQ_OPT_LEN;
	rreqOption->nIdentification = ++DSR_DEV_VAR(pstruEventDetails->nDeviceId)->nRREQIdentification;
	rreqOption->targetAddress = target;
	rreq->pstruNetworkData->nTTL = 1; //Initially hop limit = 1

	table=(DSR_RREQ_TABLE*)RREQTABLE_ALLOC();
	table->target = IP_COPY(rreqOption->targetAddress);
	table->lastRequestTime = dTime;
	table->nCount = 1;
	table->nTTL = 1;
	table->nIdentification = rreqOption->nIdentification;
	LIST_ADD_LAST(rreqTable,table);

	//Time out event
	memcpy(&pevent,pstruEventDetails,sizeof* pstruEventDetails);
	pevent.dEventTime += NonpropRequestTimeout;
	pevent.nInterfaceId = 1;
	pevent.dPacketSize = 0;
	pevent.nApplicationId = 0;
	pevent.nEventType = TIMER_EVENT;
	pevent.nPacketId = 0;
	pevent.nProtocolId = NW_PROTOCOL_DSR;
	pevent.nSubEventType = subevent_RREQ_TIMEOUT;
	pevent.pPacket = NULL;
	pevent.szOtherDetails = IP_COPY(rreqOption->targetAddress);
	table->nEventId = fnpAddEvent(&pevent);

	//Update the metrics
	DSR_DEV_VAR(nDeviceId)->dsrMetrics.rreqSent++;
	return rreq;
}
/**
This function checks if a route is present in the route cache. If not, then it retransmits 
the route request packet if the route request count is less than the DSR Max Retransmit limit.
If the RREQ count is > Max Retransmit Limit, it Empties the Send Buffer and deletes the RREQ 
entry from the RREQ table.
*/
int fn_NetSim_DSR_RREQTimeout(NetSim_EVENTDETAILS* pstruEventDetails)
{
	DSR_ROUTE_CACHE* cache;
	NETSIM_IPAddress nextHop;
	NETSIM_IPAddress targetAddress=(NETSIM_IPAddress)pstruEventDetails->szOtherDetails;
	//check for route exits or not
	if(!DSR_CHECK_ROUTE_FOUND(targetAddress,DSR_DEV_VAR(pstruEventDetails->nDeviceId),&nextHop,pstruEventDetails->dEventTime,&cache))
	{
		//No route reply received
		DSR_RREQ_TABLE* table = getRREQTable(targetAddress,DSR_DEV_VAR(pstruEventDetails->nDeviceId)->pstruRREQTable);
		if(table->nCount < DSR_MAX_REQUEST_REXMT)
		{
			//Retry sending route request
			pstruEventDetails->nEventType = NETWORK_OUT_EVENT;
			pstruEventDetails->pPacket = DSR_RETRY_RREQ(targetAddress,
				pstruEventDetails->nDeviceId,pstruEventDetails->dEventTime);
			pstruEventDetails->nSubEventType = 0;
			IP_FREE(pstruEventDetails->szOtherDetails);
			pstruEventDetails->szOtherDetails = NULL;
			fnpAddEvent(pstruEventDetails);
			return 0;
		}
		else
		{
			DSR_EMPTY_SEND_BUFFER(targetAddress,pstruEventDetails->nDeviceId);
			fn_NetSim_DSR_DeleteRREQTable(targetAddress,pstruEventDetails->nDeviceId);
		}
	}
	else
	{
		//Route reply already received. Ignore the event
	}
	IP_FREE(pstruEventDetails->szOtherDetails);
	pstruEventDetails->szOtherDetails = NULL;
	return 0;
}
/**
This function retransmits the RREQ packet.
*/
NetSim_PACKET* fn_NetSim_DSR_RetryRREQ(NETSIM_IPAddress targetAddress,
	NETSIM_ID nDeviceId,
	double dTime,
	NetSim_EVENTDETAILS* pstruEventDetails)
{
	NetSim_EVENTDETAILS pevent;
	DSR_RREQ_TABLE* rreqTable = DSR_DEV_VAR(nDeviceId)->pstruRREQTable;
	NetSim_PACKET* rreq = fn_NetSim_DSR_GenerateCtrlPacket(nDeviceId,
		0,0,/*BROADCAST*/
		dTime,
		ctrlPacket_ROUTE_REQUEST);
	DSR_OPTION_HEADER* option=(DSR_OPTION_HEADER*)calloc(1,sizeof* option);
	DSR_RREQ_OPTION* rreqOption=(DSR_RREQ_OPTION*)calloc(1,sizeof* rreqOption);
	DSR_RREQ_TABLE* table;
	rreq->pstruNetworkData->dPacketSize=DSR_RREQ_SIZE_FIXED+DSR_OPTION_HEADER_SIZE;
	rreq->pstruNetworkData->dOverhead=DSR_RREQ_SIZE_FIXED+DSR_OPTION_HEADER_SIZE;
	rreq->pstruNetworkData->dPayload=0;

	rreq->pstruNetworkData->Packet_RoutingProtocol = option;
	option->nFlowState=0;
	option->nNextHeader=NO_NEXT_HEADER;
	option->nReserved = 0;
	option->nPayloadLength=DSR_RREQ_SIZE_FIXED;
	option->optType = optType_RouteRequest;

	option->options = rreqOption;
	rreqOption->nOptionType = optType_RouteRequest;
	rreqOption->nOptDataLen = DSR_RREQ_OPT_LEN;
	rreqOption->nIdentification = ++DSR_DEV_VAR(nDeviceId)->nRREQIdentification;
	rreqOption->targetAddress = IP_COPY(targetAddress);
	rreq->pstruNetworkData->nTTL = DSR_DISCOVERY_HOP_LIMIT;

	table=getRREQTable(targetAddress,rreqTable);
	table->lastRequestTime = dTime;
	table->nCount++;
	table->nIdentification = rreqOption->nIdentification;
	table->nTTL = DSR_DISCOVERY_HOP_LIMIT;

	//Time out event
	memcpy(&pevent,pstruEventDetails,sizeof* pstruEventDetails);
	pevent.dEventTime += getRREQBackoff(table);
	pevent.dPacketSize = 0;
	pevent.nApplicationId = 0;
	pevent.nEventType = TIMER_EVENT;
	pevent.nPacketId = 0;
	pevent.nProtocolId = NW_PROTOCOL_DSR;
	pevent.nSubEventType = subevent_RREQ_TIMEOUT;
	pevent.pPacket = NULL;
	pevent.szOtherDetails = IP_COPY(rreqOption->targetAddress);
	table->nEventId = fnpAddEvent(&pevent);
	//Update the metrics
	DSR_DEV_VAR(nDeviceId)->dsrMetrics.rreqSent++;
	return rreq;
}
/**
If a RREQ to a target is before hand generated, this entry is made in the RREQ Table.
This function gets the RREQ table of the target IP if the RREQ was beforehand sent.
*/
DSR_RREQ_TABLE* getRREQTable(NETSIM_IPAddress target,DSR_RREQ_TABLE* table)
{
	while(table)
	{

		if(!IP_COMPARE(table->target,target) && table->flag==false)
			return table;
		table = LIST_NEXT(table);
	}
	return NULL;
}
/**
This gets the Backoff time of the RREQ table. The backoff time is time until when if the RREQ 
is not received, it retransmits the RREQ.
*/
double getRREQBackoff(DSR_RREQ_TABLE* table)
{
	if(!table->dBackoff)
	{
		table->dBackoff = DSR_REQUEST_PERIOD;
		return DSR_REQUEST_PERIOD;
	}
	table->dBackoff *= 2;
	if(table->dBackoff > DSR_MAX_REQUEST_PERIOD)
		table->dBackoff = DSR_MAX_REQUEST_PERIOD;
	return table->dBackoff;
}
/**
This function deletes the entry from the Route Request table
*/
int fn_NetSim_DSR_DeleteRREQTable(NETSIM_IPAddress targetAddress,NETSIM_ID nDeviceId)
{
	DSR_RREQ_TABLE* table = DSR_DEV_VAR(nDeviceId)->pstruRREQTable;
	while(table)
	{

		if(!IP_COMPARE(table->target,targetAddress))
		{
			LIST_FREE(&DSR_DEV_VAR(nDeviceId)->pstruRREQTable,table);
			break;
		}
		table = LIST_NEXT(table);
	}
	return 0;
}
/**
This function process the RREQ that a device gets.
If the target address is the device, it generates a RREP. 
If the RREQ packet contains the device IP in its address list, it drops the packet.
If the Device cache contains the route to the target, then the device replies via the route 
cache.
If no route is present, then the device forwards the RREQ.
*/
int fn_NetSim_DSR_ProcessRREQ(NetSim_EVENTDETAILS* pstruEventDetails)
{
	DSR_ROUTE_CACHE* cache;
	NetSim_PACKET* packet = pstruEventDetails->pPacket;
	DSR_OPTION_HEADER* option = (DSR_OPTION_HEADER*)packet->pstruNetworkData->Packet_RoutingProtocol;
	DSR_RREQ_OPTION* rreq = (DSR_RREQ_OPTION*)option->options;
	if (fn_NetSim_DSR_CheckEntryInRREQTable(rreq, DSR_DEV_VAR(pstruEventDetails->nDeviceId)->pstruRREQTable))
	{
		//Drop the route request packet
		fn_NetSim_Packet_FreePacket(packet);
		pstruEventDetails->pPacket = NULL;
		return 0;
	}
	else
	{
		DSR_RREQ_TABLE* table;
		table = (DSR_RREQ_TABLE*)RREQTABLE_ALLOC();
		table->target = IP_COPY(rreq->targetAddress);
		table->lastRequestTime = pstruEventDetails->dEventTime;
		table->nCount = 1;
		table->nTTL = 1;
		table->nIdentification = rreq->nIdentification;
		LIST_ADD_LAST(&DSR_DEV_VAR(pstruEventDetails->nDeviceId)->pstruRREQTable, table);
	}

	if(!IP_COMPARE(rreq->targetAddress,dsr_get_curr_ip()))
	{
		DSR_GENERATE_RREP(packet);
		//Free route request packet
		fn_NetSim_Packet_FreePacket(packet);
	}
	else
	{
		int length = DSR_RREQ_LEN(rreq);
		//Check for own ip address in address list
		if(fn_NetSim_DSR_CheckIPinIPList(rreq->address,length,dsr_get_curr_ip()))
		{
			//Drop the route request packet
			fn_NetSim_Packet_FreePacket(packet);
			pstruEventDetails->pPacket = NULL;
			return 0;
		}
		else
		{

			NETSIM_IPAddress nexthop;
			//Add entry to route request table
			DSR_RREQ_TABLE* table = RREQTABLE_ALLOC();
			table->nIdentification = rreq->nIdentification;
			table->target = IP_COPY(rreq->targetAddress);
			LIST_ADD_LAST((void**)&DSR_DEV_VAR(pstruEventDetails->nDeviceId)->pstruRREQTable, table);
			if (fn_NetSim_DSR_CheckRouteFound(rreq->targetAddress,
											  DSR_DEV_VAR(pstruEventDetails->nDeviceId),
											  &nexthop,
											  pstruEventDetails->dEventTime,
											  &cache))
			{
				//Route already present. reply using route cache
				fn_NetSim_DSR_GenerateRREPUsingRouteCache(DSR_DEV_VAR(pstruEventDetails->nDeviceId),
														  pstruEventDetails->pPacket,
														  pstruEventDetails->dEventTime,
														  pstruEventDetails);
			}
			else
			{
				//No route present forward the route request
				rreq->address = (NETSIM_IPAddress*)realloc(rreq->address, (sizeof *rreq->address)*(DSR_RREQ_LEN(rreq) + 1));
				rreq->address[DSR_RREQ_LEN(rreq)] = dsr_get_curr_ip();
				rreq->nOptDataLen += 4;
				pstruEventDetails->pPacket->pstruNetworkData->dOverhead += 4.0;
				pstruEventDetails->pPacket->pstruNetworkData->dPacketSize += 4.0;
				pstruEventDetails->pPacket->nTransmitterId = pstruEventDetails->nDeviceId;
				pstruEventDetails->pPacket->nReceiverId = 0;
				//Generate network out event
				pstruEventDetails->dEventTime += fn_NetSim_DSR_GetBroadCastJitter();
				pstruEventDetails->nEventType = NETWORK_OUT_EVENT;
				fnpAddEvent(pstruEventDetails);
				//Update the metrics
				DSR_DEV_VAR(pstruEventDetails->nDeviceId)->dsrMetrics.rreqForwarded++;
			}
		}
	}
	return 1;
}
/**
This functions checks if the IP is in the IP_List. It returns true if IP is there or else 
it returns false.
*/
bool fn_NetSim_DSR_CheckIPinIPList(NETSIM_IPAddress* ipList,int length,NETSIM_IPAddress ip)
{
	while(length--)
	{

		if(!IP_COMPARE(ipList[length],ip))
			return true;
	}
	return false;
}
/**
This function checks if the RREQ table has an entry of the target IP address. This indicates 
that the particular device has already initiated a RREQ to the target.
*/
bool fn_NetSim_DSR_CheckEntryInRREQTable(DSR_RREQ_OPTION* rreq,DSR_RREQ_TABLE* table)
{
	while(table)
	{

		if(table->nIdentification == rreq->nIdentification && !IP_COMPARE(table->target,rreq->targetAddress))
			return true;
		table = LIST_NEXT(table);
	}
	return false;
}


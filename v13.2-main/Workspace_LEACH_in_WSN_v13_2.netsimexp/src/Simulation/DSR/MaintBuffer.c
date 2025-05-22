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
Whenever a packet is to be sent to the target and the device has the next Hop IP,
the packet is added to the MaintBuffer and retained there until and acknowledge is
received or a timeout happens in which case the packets are retransmitted.
*/
int fn_NetSim_DSR_AddToMaintBuffer(NETSIM_ID nDeviceId,
	NetSim_PACKET* pstruPacket,
	double dTime)
{
	NetSim_PACKET* packet = fn_NetSim_Packet_CopyPacket(pstruPacket);
	NetSim_EVENTDETAILS pevent;
	//Get the Maint buffer
	DSR_MAINT_BUFFER** maintBuffer = &(DSR_DEV_VAR(nDeviceId)->pstruMaintBuffer);
	DSR_MAINT_BUFFER* buffer=*maintBuffer;
	if(!*maintBuffer)
	{
		*maintBuffer = MAINTBUFFER_ALLOC();
		(*maintBuffer)->dAckTime =  dTime + MAINT_HOLD_OFF_TIME;
		(*maintBuffer)->nextHop = packet->nReceiverId;
		(*maintBuffer)->source = packet->nSourceId;
		(*maintBuffer)->dest = get_first_dest_from_packet(packet);
		(*maintBuffer)->packetList = packet;
		memcpy(&pevent,pstruEventDetails,sizeof* pstruEventDetails);
		pevent.dEventTime = (*maintBuffer)->dAckTime;
		pevent.dPacketSize = 0;
		pevent.nApplicationId = 0;
		pevent.nEventType = TIMER_EVENT;
		pevent.nPacketId = 0;
		pevent.nProtocolId = NW_PROTOCOL_DSR;
		pevent.nSegmentId =0;
		pevent.nSubEventType = subevent_MAINT_TIMEOUT;
		pevent.pPacket = NULL;
		pevent.szOtherDetails = calloc(4,sizeof(NETSIM_ID));
		*((NETSIM_ID*)pevent.szOtherDetails) = (*maintBuffer)->source;
		*((NETSIM_ID*)(pevent.szOtherDetails) + 1) = (*maintBuffer)->dest;
		*((NETSIM_ID*)(pevent.szOtherDetails) + 2) = (*maintBuffer)->nextHop;
		fnpAddEvent(&pevent);
		return 1;
	}
	while(buffer)
	{
		if(buffer->nextHop == packet->nReceiverId &&
			buffer->dest == get_first_dest_from_packet(packet) &&
			buffer->source == packet->nSourceId)
		{
			if(buffer->packetList)
			{
				NetSim_PACKET* packetList = buffer->packetList;
				while(packetList->pstruNextPacket)
					packetList = packetList->pstruNextPacket;
				packetList->pstruNextPacket = packet;
			}
			else
			{
				buffer->dAckTime = dTime+MAINT_HOLD_OFF_TIME;
				buffer->packetList = packet;
				memcpy(&pevent,pstruEventDetails,sizeof* pstruEventDetails);
				pevent.nInterfaceId = 1;
				pevent.dEventTime = buffer->dAckTime;
				pevent.dPacketSize = 0;
				pevent.nApplicationId = 0;
				pevent.nEventType = TIMER_EVENT;
				pevent.nPacketId = 0;
				pevent.nProtocolId = NW_PROTOCOL_DSR;
				pevent.nSegmentId =0;
				pevent.nSubEventType = subevent_MAINT_TIMEOUT;
				pevent.pPacket = NULL;
				pevent.szOtherDetails = calloc(4,sizeof(NETSIM_ID));
				*((NETSIM_ID*)pevent.szOtherDetails) = buffer->source;
				*((NETSIM_ID*)(pevent.szOtherDetails) + 1) = buffer->dest;
				*((NETSIM_ID*)(pevent.szOtherDetails) + 2) = buffer->nextHop;
				fnpAddEvent(&pevent);
			}
			return 2;
		}
		buffer=LIST_NEXT(buffer);
	}
	buffer = MAINTBUFFER_ALLOC();
	buffer->dAckTime = dTime+MAINT_HOLD_OFF_TIME;
	buffer->packetList = packet;
	buffer->dest = get_first_dest_from_packet(packet);
	buffer->nextHop = packet->nReceiverId;
	buffer->source = packet->nSourceId;
	memcpy(&pevent,pstruEventDetails,sizeof* pstruEventDetails);
	pevent.dEventTime = buffer->dAckTime;
	pevent.dPacketSize = 0;
	pevent.nApplicationId = 0;
	pevent.nEventType = TIMER_EVENT;
	pevent.nPacketId = 0;
	pevent.nProtocolId = NW_PROTOCOL_DSR;
	pevent.nSegmentId =0;
	pevent.nSubEventType = subevent_MAINT_TIMEOUT;
	pevent.pPacket = NULL;
	pevent.szOtherDetails = calloc(4,sizeof(NETSIM_ID));
	*((NETSIM_ID*)pevent.szOtherDetails) = buffer->source;
	*((NETSIM_ID*)(pevent.szOtherDetails) + 1) = buffer->dest;
	*((NETSIM_ID*)(pevent.szOtherDetails) + 2) = buffer->nextHop;
	fnpAddEvent(&pevent);
	LIST_ADD_LAST(maintBuffer,buffer);
	return 3;
}
/**
This is called when DSR Maintenance Timeout happens. It retransmits the packets if acknowledgemnet 
is not received and the transmit count is less the DSR MaxMaintenance Retransmit limit.
Else it empties the buffer.
*/
int fn_NetSim_DSR_MaintTimeout(NetSim_EVENTDETAILS* pstruEventDetails)
{
	DSR_MAINT_BUFFER* maintBuffer = DSR_DEV_VAR(pstruEventDetails->nDeviceId)->pstruMaintBuffer;
	NETSIM_ID source = *((NETSIM_ID*)pstruEventDetails->szOtherDetails);
	NETSIM_ID dest = *((NETSIM_ID*)pstruEventDetails->szOtherDetails+1);
	NETSIM_ID nextHop = *((NETSIM_ID*)pstruEventDetails->szOtherDetails+2);
	free((NETSIM_ID*)pstruEventDetails->szOtherDetails);
	pstruEventDetails->szOtherDetails = NULL;
	while(maintBuffer)
	{
		if(maintBuffer->nextHop == nextHop &&
			maintBuffer->source == source &&
			maintBuffer->dest == dest)
		{
			if(maintBuffer->dAckTime <= pstruEventDetails->dEventTime)
			{
				if(maintBuffer->packetList)
				{
					if(maintBuffer->count<DSR_MaxMaintRexmt)
					{
						maintBuffer->count++;
						DSR_RETRANSMIT_BUFFER(maintBuffer);
					}
					else
					{
						maintBuffer->count=0;
						//Generate route error
						DSR_GENERATE_RERR(maintBuffer);
						DSR_EMPTY_MAINT_BUFFER(pstruEventDetails->nDeviceId,nextHop);
					}
					//update the metrics
					DSR_DEV_VAR(pstruEventDetails->nDeviceId)->dsrMetrics.routeBreak++;
				}
				else
				{
					//ignore the event. All packet is acked.
				}
			}
			else
			{
				//ignore the event
			}
			break;
		}
		maintBuffer = LIST_NEXT(maintBuffer);
	}
	return 1;
}
/**
This function retransmits the packets in the DSR Maintanence Buffer.
*/
int fn_NetSim_DSR_RetransmitBuffer(DSR_MAINT_BUFFER* maintBuffer,NetSim_EVENTDETAILS* pstruEventDetails)
{
	NetSim_EVENTDETAILS pevent;
	memcpy(&pevent,pstruEventDetails,sizeof* pstruEventDetails);
	while(maintBuffer->packetList)
	{
		NetSim_PACKET* packet = maintBuffer->packetList;
		maintBuffer->packetList = maintBuffer->packetList->pstruNextPacket;
		packet->pstruNextPacket = NULL;

		pevent.dPacketSize = fnGetPacketSize(packet);
		if(packet->pstruAppData)
		{
			pevent.nApplicationId = packet->pstruAppData->nApplicationId;
			pevent.nSegmentId = packet->pstruAppData->nSegmentId;
		}
		else
		{
			pevent.nApplicationId = 0;
			pevent.nSegmentId = 0;
		}
		packet->pstruNetworkData->szNextHopIp=NULL;
		pevent.nEventType = NETWORK_OUT_EVENT;
		pevent.nPacketId = packet->nPacketId;
		pevent.nProtocolId = NW_PROTOCOL_DSR;
		pevent.nSubEventType = 0;
		pevent.pPacket = packet;
		pevent.szOtherDetails = NULL;
		fnpAddEvent(&pevent);
	}
	return 1;
}
/**
This function Empties the DSR Maintenance Buffer.
*/
int fn_NetSim_DSR_EmptyMaintBuffer(NETSIM_ID nDeviceId,NETSIM_ID nextHop)
{
	DSR_MAINT_BUFFER* maintBuffer = DSR_DEV_VAR(nDeviceId)->pstruMaintBuffer;
	while(maintBuffer)
	{
		if(maintBuffer->nextHop == nextHop)
		{
			while(maintBuffer->packetList)
			{
				NetSim_PACKET* packet = maintBuffer->packetList;
				maintBuffer->packetList = maintBuffer->packetList->pstruNextPacket;
				packet->pstruNextPacket = NULL;
				fn_NetSim_Packet_FreePacket(packet);
			}
		}
		maintBuffer = LIST_NEXT(maintBuffer);
	}
	return 0;
}
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
This function adds the packet to send buffer.
*/
bool fn_NetSim_DSR_AddToSendBuffer(DSR_SEND_BUFFER** sendBuffer,NetSim_PACKET* packet,double dTime)
{
	NETSIM_IPAddress dest = packet->pstruNetworkData->szDestIP;
	DSR_SEND_BUFFER* temp = *sendBuffer;
	bool flag=true;
	while(temp)
	{
		if(!IP_COMPARE(dest,temp->target))
		{
			flag = false;
			break;
		}
		temp = LIST_NEXT(temp);
	}
	if(!temp)
		temp = (DSR_SEND_BUFFER*)SENDBUFFER_ALLOC();
	if(!temp->packet)
	{
		temp->dTime = dTime;
		temp->packet = packet;
		temp->target = dest;
	}
	else
	{
		NetSim_PACKET* temp_packet=temp->packet;
		while(temp_packet->pstruNextPacket)
			temp_packet = temp_packet->pstruNextPacket;
		temp_packet->pstruNextPacket = packet;
	}
	if(flag)
		LIST_ADD_LAST((void**)sendBuffer,temp);
	return flag;
}
/**
This function Empties the send buffer.
*/
void fn_NetSim_DSR_EmptySendBuffer(NETSIM_IPAddress targetAddress,NETSIM_ID nDeviceId)
{
	DSR_SEND_BUFFER* sendBuffer=DSR_DEV_VAR(nDeviceId)->pstruSendBuffer;
	while(sendBuffer)
	{
		if(!IP_COMPARE(sendBuffer->target,targetAddress))
		{
			while(sendBuffer->packet)
			{
				NetSim_PACKET* temp = sendBuffer->packet;
				sendBuffer->packet = temp->pstruNextPacket;
				temp->pstruNextPacket = NULL;
				fn_NetSim_Packet_FreePacket(temp);
				//Update the metrics
				DSR_DEV_VAR(nDeviceId)->dsrMetrics.packetDropped++;
			}
			IP_FREE(sendBuffer->target);
			LIST_FREE(&DSR_DEV_VAR(nDeviceId)->pstruSendBuffer,sendBuffer);
			break;
		}
		sendBuffer = LIST_NEXT(sendBuffer);
	}
}
/**
This function checks if the route cache has the route to the target and transmits the packet 
if it has.
*/
void fn_NetSim_DSR_CheckSendBuffer(NETSIM_ID nDeviceId,double dTime)
{
	DSR_SEND_BUFFER* sendBuffer=DSR_DEV_VAR(nDeviceId)->pstruSendBuffer;
	DSR_ROUTE_CACHE* routeCache=DSR_DEV_VAR(nDeviceId)->pstruRouteCache;
	while(sendBuffer)
	{
		int ntxflag=0;
		NETSIM_IPAddress target = sendBuffer->target;
		while(routeCache)
		{
			unsigned int nLoop;
			for(nLoop=0;nLoop<routeCache->nLength;nLoop++)
			{
				if(!IP_COMPARE(target,routeCache->address[nLoop]))
				{
					//Route found
					DSR_TRANSMIT_SEND_BUFFER(sendBuffer,
						nDeviceId,dTime);
					ntxflag=1;
					break;
				}
			}
			routeCache = LIST_NEXT(routeCache);
		}
		if(ntxflag)
		{
			DSR_SEND_BUFFER* temp = sendBuffer;
			sendBuffer=LIST_NEXT(sendBuffer);
			DSR_EMPTY_SEND_BUFFER(temp->target,nDeviceId);
			continue;
		}
		sendBuffer = LIST_NEXT(sendBuffer);
	}
}
/**
This function transmits the packets from sent buffer
*/
int fn_NetSim_DSR_TransmitPacketFromSendBuffer(DSR_SEND_BUFFER* sendBuffer,
	NETSIM_ID nDeviceId,
	double dTime)
{
	NetSim_EVENTDETAILS eventDetails;
	memset(&eventDetails,0,sizeof*(&eventDetails));
	while(sendBuffer->packet)
	{
		NetSim_PACKET* packet = sendBuffer->packet;
		sendBuffer->packet = packet->pstruNextPacket;
		packet->pstruNextPacket = NULL;
		eventDetails.dEventTime = dTime;
		eventDetails.dPacketSize = fnGetPacketSize(packet);
		if(packet->pstruAppData)
		{
			eventDetails.nApplicationId = packet->pstruAppData->nApplicationId;
			eventDetails.nSegmentId = packet->pstruAppData->nSegmentId;
		}
		else
		{
			eventDetails.nApplicationId = 0;
			eventDetails.nSegmentId = 0;
		}
		eventDetails.nDeviceId = nDeviceId;
		eventDetails.nDeviceType = NETWORK->ppstruDeviceList[nDeviceId-1]->nDeviceType;
		eventDetails.nEventType = NETWORK_OUT_EVENT;
		eventDetails.nInterfaceId = 1;
		eventDetails.nPacketId = packet->nPacketId;
		eventDetails.nProtocolId = fn_NetSim_Stack_GetNWProtocol(nDeviceId);
		eventDetails.nSubEventType = 0;
		eventDetails.pPacket = packet;
		fnpAddEvent(&eventDetails);
	}
	return 0;
}
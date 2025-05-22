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
This function is called in the Network Out Event for routing of the packet.<br/>

Whenever a packet is to be routed to a destination and the device route cache doesnt have
a route to the target, the Packet is added to the Send buffer and a Route Discovery is 
iitiated.<br/>

If the Device route cache has a route to the target, then the packet is added to the Maintanence
Buffer and the count of packets originated is incremented.

*/
int fn_NetSim_DSR_GeneralPacketProcessing(NetSim_EVENTDETAILS* pstruEventDetails)
{
	NETSIM_IPAddress nextHop;
	NETSIM_ID nInterface;
	DSR_ROUTE_CACHE* cache=NULL;
	DSR_DEVICE_VAR* pstruDevVar = DSR_DEV_VAR(pstruEventDetails->nDeviceId);
	pstruEventDetails->pPacket->pstruNetworkData->szGatewayIP = IP_COPY(dsr_get_curr_ip());
	//LEACH
	fn_NetSim_LEACH_GetNextHop(pstruEventDetails);
	return 1;
	//LEACH
	if(DSR_CHECK_ROUTE_FOUND(pstruEventDetails->pPacket->pstruNetworkData->szDestIP,
		pstruDevVar,&nextHop,pstruEventDetails->dEventTime,&cache))
	{
		//Route found
		DSR_ADD_SRC_ROUTE(pstruEventDetails->pPacket,cache);
		DSR_ADD_ACK_REQUEST(pstruEventDetails->pPacket);
		IP_FREE(pstruEventDetails->pPacket->pstruNetworkData->szNextHopIp);
		pstruEventDetails->pPacket->pstruNetworkData->szNextHopIp = IP_COPY(nextHop);
		pstruEventDetails->pPacket->pstruNetworkData->szGatewayIP = IP_COPY(dsr_get_curr_ip());
		pstruEventDetails->pPacket->nTransmitterId = pstruEventDetails->nDeviceId;
		pstruEventDetails->pPacket->nReceiverId = fn_NetSim_Stack_GetDeviceId_asIP(nextHop,&nInterface);
		DSR_ADD_TO_MAINT_BUFFER(pstruEventDetails->nDeviceId,pstruEventDetails->pPacket,pstruEventDetails->dEventTime);
		DSR_DEV_VAR(pstruEventDetails->nDeviceId)->dsrMetrics.packetTransmitted++;
		if(pstruEventDetails->pPacket->nSourceId == pstruEventDetails->nDeviceId)
			DSR_DEV_VAR(pstruEventDetails->nDeviceId)->dsrMetrics.packetOrginated++;
		if(DSR_DEV_VAR(pstruEventDetails->nDeviceId)->AckType == LINK_LAYER_ACK)
			pstruEventDetails->pPacket->ReceiveAckNotification = fn_NetSim_DSR_LinkLayerAck;
	}
	else
	{
		//Add packet to send buffer
		if(DSR_ADD_TO_SEND_BUFFER(&pstruDevVar->pstruSendBuffer,
			pstruEventDetails->pPacket,
			pstruEventDetails->dEventTime))
		{
			//Initiate Route discovery process
			NetSim_PACKET* packet = DSR_INIT_RREQ(pstruEventDetails->pPacket->pstruNetworkData->szDestIP,
				&pstruDevVar->pstruRREQTable,
				pstruEventDetails->nDeviceId,
				pstruEventDetails->dEventTime);
			pstruEventDetails->pPacket = packet;
			pstruEventDetails->dPacketSize = packet->pstruNetworkData->dPacketSize;
			pstruEventDetails->nApplicationId = 0;
			pstruEventDetails->nPacketId = 0;
		}
		else
		{
			//RREQ already sent
			pstruEventDetails->pPacket = NULL;//Don't transmit the packet
		}
	}
	return 1;
}

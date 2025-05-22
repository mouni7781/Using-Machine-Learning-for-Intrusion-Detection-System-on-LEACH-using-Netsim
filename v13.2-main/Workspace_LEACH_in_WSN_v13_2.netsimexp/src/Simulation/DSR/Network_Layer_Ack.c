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
/**
This function adds the Acknowledge request option to the packet.
*/
int fn_NetSim_DSR_Add_Ack_request_Option(NetSim_PACKET* packet,
	NetSim_EVENTDETAILS* pstruEventDetails)
{
	DSR_OPTION_HEADER* option = (DSR_OPTION_HEADER*)packet->pstruNetworkData->Packet_RoutingProtocol;
	static unsigned int nIdentification=1; 
	if(DSR_DEV_VAR(pstruEventDetails->nDeviceId)->AckType == NETWORK_LAYER_ACK)
	{
		if(!option->ackRequestOption)
			option->ackRequestOption = (DSR_ACK_REQ_OPTION*)calloc(1,sizeof* option->ackRequestOption);
		option->ackRequestOption->nOptionType = optType_AckRequest;
		option->ackRequestOption->nIdentification = nIdentification++;
		packet->pstruNetworkData->dPacketSize+=DSR_ACK_REQUEST_LEN;
		packet->pstruNetworkData->dOverhead+=DSR_ACK_REQUEST_LEN;
	}
	else
	{
		if(option->ackRequestOption)
		{
			option->ackRequestOption = NULL;
			packet->pstruNetworkData->dPacketSize-=DSR_ACK_REQUEST_LEN;
			packet->pstruNetworkData->dOverhead-=DSR_ACK_REQUEST_LEN;
		}
	}
	return 1;
}
/**
If a packet has an acknowledge request option, this function generates the acknowledge
and transmits it.
*/
int fn_NetSim_DSR_Process_AckRequestOption(NetSim_PACKET* packet,NetSim_EVENTDETAILS* pstruEventDetails)
{
	DSR_OPTION_HEADER* option = (DSR_OPTION_HEADER*)packet->pstruNetworkData->Packet_RoutingProtocol;
	if(option->ackRequestOption)
	{
		NetSim_EVENTDETAILS pevent;
		DSR_ACK_REQ_OPTION* ackRequest = option->ackRequestOption;
		NetSim_PACKET* ppacket = fn_NetSim_DSR_GenerateCtrlPacket(pstruEventDetails->nDeviceId,
			packet->nTransmitterId,
			packet->nTransmitterId,
			pstruEventDetails->dEventTime,
			ctrlPacket_ACK);
		option = (DSR_OPTION_HEADER*)calloc(1,sizeof* option);
		DSR_ACK_OPTION* ack = (DSR_ACK_OPTION*)calloc(1,sizeof* ack);
		ppacket->pstruNetworkData->Packet_RoutingProtocol = option;
		option->options = ack;
		option->nNextHeader = NO_NEXT_HEADER;
		option->optType = optType_Ack;
		option->nPayloadLength = DSR_ACK_OPTION_LEN;

		ack->DestAddress = dsr_get_dev_ip(packet->nTransmitterId);
		ack->nIdentification = ackRequest->nIdentification;
		ack->nOptionDataLen = DSR_ACK_OPTION_LEN;
		ack->nOptionType = optType_Ack;
		ack->sourceAddress = dsr_get_curr_ip();
		ppacket->pstruNetworkData->dOverhead = DSR_ACK_OPTION_LEN+DSR_OPTION_HEADER_SIZE;
		ppacket->pstruNetworkData->dPacketSize = DSR_ACK_OPTION_LEN+DSR_OPTION_HEADER_SIZE;
		ppacket->pstruNetworkData->nTTL = 2;
		//Transmit ack
		memcpy(&pevent,pstruEventDetails,sizeof* pstruEventDetails);
		pevent.dPacketSize = DSR_ACK_OPTION_LEN+DSR_OPTION_HEADER_SIZE;
		pevent.nApplicationId = 0;
		pevent.nEventType = NETWORK_OUT_EVENT;
		pevent.nPacketId = 0;
		pevent.nProtocolId = fn_NetSim_Stack_GetNWProtocol(pevent.nDeviceId);
		pevent.nSegmentId = 0;
		pevent.nSubEventType = 0;
		pevent.pPacket = ppacket;
		fnpAddEvent(&pevent);
	}
	return 1;
}
/**
Whenever an acknowlegement is received, the DSR Maintenance Buffer is Emptied.
*/
int fn_NetSim_DSR_ProcessAckOption(NetSim_EVENTDETAILS* pstruEventDetails)
{
	NetSim_PACKET* packet = pstruEventDetails->pPacket;
	NETSIM_ID nSource = packet->nSourceId;
	DSR_EMPTY_MAINT_BUFFER(pstruEventDetails->nDeviceId,nSource);
	fn_NetSim_Packet_FreePacket(packet);
	pstruEventDetails->pPacket=NULL;
	return 1;
}



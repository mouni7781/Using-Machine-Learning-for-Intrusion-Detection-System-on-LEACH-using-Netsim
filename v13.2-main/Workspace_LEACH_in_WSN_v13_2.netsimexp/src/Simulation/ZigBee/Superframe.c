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
# include "main.h"
# include "802_15_4.h"
/** 
The format of the superframe is defined by the coordinator. The superframe is bounded by network beacons 
sent by the coordinator and is divided into 16 equally sized slots. Optionally, the superframe can have an 
active and an inactive portion.
During the inactive portion, the coordinator may enter a low-power mode. The beacon frame is transmitted in 
the first slot of each superframe. If a coordinator does not wish to use a superframe structure, it will 
turn off the beacon transmissions.The beacons are used to synchronize the attached devices, to identify 
the PAN, and to describe the structure of the superframes.
*/
int fn_NetSim_Zigbee_SuperFrameInitialization(SUPERFRAME** ppstruSuperFrame,NETSIM_ID nGlobalPANCoordinatorId,METRICS** pstruMetrics)
{
	SUPERFRAME* pstruSuperFrame=*ppstruSuperFrame;
	static int nBeaconId;
	NETSIM_ID nDeviceId;
	double dTime;
	NetSim_PACKET *pstruPacket;
	int nCFPStartingSlot = 16;
	NETSIM_ID nLink_Id, nConnectionID, nConnectionPortID,nLoop;

	nLink_Id = fn_NetSim_Stack_GetConnectedDevice(pstruEventDetails->nDeviceId,pstruEventDetails->nInterfaceId,&nConnectionID,&nConnectionPortID);

	for(nLoop=1; nLoop<=NETWORK->ppstruNetSimLinks[nLink_Id-1]->puniDevList.pstruMP2MP.nConnectedDeviceCount; nLoop++)
	{
		nDeviceId = NETWORK->ppstruNetSimLinks[nLink_Id-1]->puniDevList.pstruMP2MP.anDevIds[nLoop-1];
		if(WSN_MAC(nDeviceId)->nNodeStatus != OFF)
			WSN_MAC(nDeviceId)->nNodeStatus = IDLE;
		if(WSN_PHY(nDeviceId)->nRadioState != RX_OFF)
			ZIGBEE_CHANGERADIOSTATE(nDeviceId, WSN_PHY(nDeviceId)->nRadioState, RX_ON_IDLE);
	}

	if(!pstruSuperFrame)
	{
		*ppstruSuperFrame = (SUPERFRAME*)fnpAllocateMemory(1,sizeof(SUPERFRAME));
		pstruSuperFrame=*ppstruSuperFrame;
		pstruSuperFrame->nSuperFrameId = 1;
		pstruSuperFrame->dSuperFrameLength = WSN_MAC(nGlobalPANCoordinatorId)->dBaseSuperFrameDuration*pow(2,WSN_MAC(nGlobalPANCoordinatorId)->nMacBeaconOrder)*1000;
		pstruSuperFrame->dActivePeriodLength = WSN_MAC(nGlobalPANCoordinatorId)->dBaseSuperFrameDuration* pow(2,WSN_MAC(nGlobalPANCoordinatorId)->nMacSuperframeOrder) * 1000;
		pstruSuperFrame->dInactivePeriodLength = pstruSuperFrame->dSuperFrameLength - pstruSuperFrame->dActivePeriodLength;
		pstruSuperFrame->dSuperFrameStartTime = 0;
		pstruSuperFrame->dTimeSlotLength = pstruSuperFrame->dActivePeriodLength/16.0;
		pstruSuperFrame->nTimeSlotCount = 16;
		pstruSuperFrame->dBeaconLength = pstruSuperFrame->dTimeSlotLength;
		pstruSuperFrame->dCAPLength = pstruSuperFrame->dActivePeriodLength - pstruSuperFrame->dBeaconLength;
		pstruSuperFrame->dCFPLength = 0;
	}
	else
	{
		pstruSuperFrame->pstruNextSuperFrame = (SUPERFRAME*)fnpAllocateMemory(1,sizeof(SUPERFRAME));
		pstruSuperFrame->pstruNextSuperFrame->nSuperFrameId = pstruSuperFrame->nSuperFrameId+1;
		pstruSuperFrame->pstruNextSuperFrame->dSuperFrameLength = pstruSuperFrame->dSuperFrameLength;
		pstruSuperFrame->pstruNextSuperFrame->dActivePeriodLength = pstruSuperFrame->dActivePeriodLength;
		pstruSuperFrame->pstruNextSuperFrame->dInactivePeriodLength = pstruSuperFrame->dInactivePeriodLength;
		pstruSuperFrame->pstruNextSuperFrame->dSuperFrameStartTime = pstruSuperFrame->dSuperFrameStartTime + pstruSuperFrame->dSuperFrameLength;
		pstruSuperFrame->pstruNextSuperFrame->nTimeSlotCount = 16;
		pstruSuperFrame->pstruNextSuperFrame->dTimeSlotLength = pstruSuperFrame->dTimeSlotLength;
		pstruSuperFrame->pstruNextSuperFrame->dBeaconLength = pstruSuperFrame->dBeaconLength;
		pstruSuperFrame->pstruNextSuperFrame->dCAPLength = pstruSuperFrame->dActivePeriodLength - pstruSuperFrame->dBeaconLength;
		pstruSuperFrame->pstruNextSuperFrame->dCFPLength = 0.0;

		//Update the CAP length and CFP length
		pstruSuperFrame->pstruNextSuperFrame->dCAPLength = pstruSuperFrame->pstruNextSuperFrame->dCAPLength - (16-(double)nCFPStartingSlot)*pstruSuperFrame->pstruNextSuperFrame->dTimeSlotLength;
		pstruSuperFrame->pstruNextSuperFrame->dCFPLength = (16-(double)nCFPStartingSlot)*pstruSuperFrame->pstruNextSuperFrame->dTimeSlotLength;
		pstruSuperFrame = pstruSuperFrame->pstruNextSuperFrame;
		*ppstruSuperFrame = pstruSuperFrame;
	}
	pstruMetrics[pstruEventDetails->nDeviceId-1]->pstruIEEE802_15_4_Metrics->nBeaconTransmitted++;
	pstruMetrics[pstruEventDetails->nDeviceId-1]->pstruIEEE802_15_4_Metrics->dBeaconTime += pstruSuperFrame->dBeaconLength;
	pstruMetrics[pstruEventDetails->nDeviceId-1]->pstruIEEE802_15_4_Metrics->dCAPTime += pstruSuperFrame->dCAPLength;
	pstruMetrics[pstruEventDetails->nDeviceId-1]->pstruIEEE802_15_4_Metrics->dCFPTime += pstruSuperFrame->dCFPLength;
	//Set the superframe status = Beacon transmission
	pstruSuperFrame->nSuperFrameStatus = BEACONTRANSMISSIONMODE;

	/*** Preparing details for beacon ***/
	pstruSuperFrame->pstruBeacon = (BEACONFRAME*)fnpAllocateMemory(1,sizeof(BEACONFRAME));
	/*{
	_declspec(dllimport) int memcheckcount;
	_declspec(dllimport) void* memcheck[50];
	memcheck[memcheckcount++]=pstruSuperFrame->pstruBeacon;
	}*/
	pstruSuperFrame->pstruBeacon->nBeaconId = ++nBeaconId;
	pstruSuperFrame->pstruBeacon->nSuperFrameId = pstruSuperFrame->nSuperFrameId;
	pstruSuperFrame->pstruBeacon->nBeaconTime = (int)(pstruSuperFrame->dSuperFrameStartTime);
	pstruSuperFrame->pstruBeacon->dPayload = 5;
	pstruSuperFrame->pstruBeacon->dOverhead = 29;
	pstruSuperFrame->pstruBeacon->dFrameSize = pstruSuperFrame->pstruBeacon->dOverhead + pstruSuperFrame->pstruBeacon->dPayload;

	pstruPacket = fn_NetSim_Packet_CreatePacket(2);
	pstruPacket->nPacketType = PacketType_Control;
	pstruPacket->nControlDataType = BEACON_FRAME;
	pstruPacket->nSourceId = nGlobalPANCoordinatorId;
	add_dest_to_packet(pstruPacket, 0);
	pstruPacket->dEventTime = pstruSuperFrame->dSuperFrameStartTime;
	pstruPacket->nTransmitterId = nGlobalPANCoordinatorId;
	pstruPacket->nReceiverId = 0;
	pstruPacket->pstruMacData->dArrivalTime = pstruSuperFrame->dSuperFrameStartTime;
	pstruPacket->pstruMacData->dStartTime = pstruSuperFrame->dSuperFrameStartTime;
	pstruPacket->pstruMacData->dEndTime = pstruSuperFrame->dSuperFrameStartTime;
	pstruPacket->pstruMacData->dPayload = pstruSuperFrame->pstruBeacon->dPayload;
	pstruPacket->pstruMacData->dOverhead = pstruSuperFrame->pstruBeacon->dPayload;
	pstruPacket->pstruMacData->dPacketSize = pstruPacket->pstruMacData->dPayload + pstruPacket->pstruMacData->dOverhead;
	pstruPacket->pstruMacData->szSourceMac = fn_NetSim_Stack_GetMacAddressFromIP(fn_NetSim_Stack_GetFirstIPAddressAsId(pstruEventDetails->nDeviceId,0));
	pstruPacket->pstruMacData->szDestMac = BROADCAST_MAC;
	pstruPacket->pstruMacData->nMACProtocol = MAC_PROTOCOL_IEEE802_15_4;
	pstruPacket->pstruMacData->Packet_MACProtocol = fnpAllocateMemory(1,sizeof(IEEE802_15_4_HEADER));
	WSN_MAC_HEADER(pstruPacket)->nAckRequestFlag = 0;
	WSN_MAC_HEADER(pstruPacket)->pstruBeaconFrame =	pstruSuperFrame->pstruBeacon;
	pstruEventDetails->dPacketSize = pstruPacket->pstruMacData->dPacketSize;
	pstruEventDetails->nApplicationId = 0;
	pstruEventDetails->nDeviceType = PANCOORDINATOR;
	pstruEventDetails->nPacketId = nBeaconId;
	pstruEventDetails->nInterfaceId = 1;
	pstruEventDetails->nEventType = PHYSICAL_OUT_EVENT;
	pstruEventDetails->nSubEventType = 0;
	pstruEventDetails->nProtocolId = MAC_PROTOCOL_IEEE802_15_4;
	pstruEventDetails->pPacket = pstruPacket;
	fnpAddEvent(pstruEventDetails);

	dTime = pstruEventDetails->dEventTime;
	//To trigger the Next superframe event
	pstruEventDetails->dPacketSize = pstruPacket->pstruMacData->dPacketSize;
	pstruEventDetails->nApplicationId = 0;
	pstruEventDetails->nDeviceType = PANCOORDINATOR;
	pstruEventDetails->nPacketId = nBeaconId;
	pstruEventDetails->dEventTime = pstruSuperFrame->dSuperFrameStartTime + pstruSuperFrame->dSuperFrameLength + 1;
	pstruEventDetails->nEventType = TIMER_EVENT;
	pstruEventDetails->nDeviceId = nGlobalPANCoordinatorId;
	pstruEventDetails->nProtocolId = MAC_PROTOCOL_IEEE802_15_4;
	pstruEventDetails->nSubEventType = SUPERFRAME_EVENT;
	pstruEventDetails->pPacket = NULL;
	fnpAddEvent(pstruEventDetails);

	//pstruEventDetails->dPacketSize = pstruPacket->pstruMacData->dPacketSize;
	pstruEventDetails->nApplicationId = 0;
	pstruEventDetails->nDeviceType = PANCOORDINATOR;
	pstruEventDetails->nPacketId = nBeaconId;
	pstruEventDetails->nDeviceId = nGlobalPANCoordinatorId;
	pstruEventDetails->dEventTime = dTime + pstruSuperFrame->dTimeSlotLength;
	pstruEventDetails->nEventType = TIMER_EVENT;
	pstruEventDetails->nSubEventType = BEACON_TRANSMISSION_END;
	pstruEventDetails->nProtocolId = MAC_PROTOCOL_IEEE802_15_4;
	pstruEventDetails->pPacket = NULL;
	fnpAddEvent(pstruEventDetails);
	return 0;
}


/************************************************************************************
* Copyright (C) 2020                                                                *
* TETCOS, Bangalore. India                                                          *
*                                                                                   *
* Tetcos owns the intellectual property rights in the Product and its content.      *
* The copying, redistribution, reselling or publication of any or all of the        *
* Product or its content without express prior written consent of Tetcos is         *
* prohibited. Ownership and / or any other right relating to the software and all   *
* intellectual property rights therein shall remain at all times with Tetcos.       *
*                                                                                   *
* Author:    Shashi Kant Suman						                                *
*                                                                                   *
* ---------------------------------------------------------------------------------*/
#include "main.h"
#include "802_15_4.h"
#include "Animation.h"
#include "Watchdog.h"

int fn_NetSim_Zigbee_FreePacket_F(NetSim_PACKET* pstruPacket);

static double zigbee_get_link_quality(NETSIM_ID trx, NETSIM_ID tri,
									  NETSIM_ID rx, NETSIM_ID ri)
{
	double p = _propagation_get_received_power_dbm(find_propagation_info(
												  trx, tri,
												  rx, ri),ldEventTime);
	assert(tri == 1);
	double rs = MW_TO_DBM(WSN_PHY(trx)->dReceiverSensivity);
	//make positive for easy calculation
	rs *= -1;
	p *= -1;
	if (rs < p)
		return 0.0;
	else if (p < 0)
		return 1.0;
	else
		return (1.0 - (1.0 / rs)*p);
}

/** This function is called by NetworkStack.dll, whenever the event gets triggered
inside the NetworkStack.dll.
It includes MAC_OUT,MAC_IN,PHY_OUT,PHY_IN and TIMER_EVENT. */
_declspec(dllexport) int fn_NetSim_Zigbee_Run()
{
	int nSub_Event_Type;
	NETSIM_ID nDeviceId, nInterfaceId;
	nEventType=pstruEventDetails->nEventType;			/* Get the EventType from Event details */
	nSub_Event_Type=pstruEventDetails->nSubEventType;	/* Get the sub EventType from Event details*/
	nDeviceId = pstruEventDetails->nDeviceId;
	nInterfaceId = pstruEventDetails->nInterfaceId;

	/*Check  event type*/
	switch(nEventType)	
	{	
	case MAC_OUT_EVENT:
		{
			NetSim_PACKET* pstruPacket=NULL;
			double dEventTime;
			dEventTime = pstruEventDetails->dEventTime;

			if(WSN_MAC(nDeviceId)->pstruBuffer)
			{
				pstruPacket = WSN_MAC(nDeviceId)->pstruBuffer;
			}
			else if(fn_NetSim_GetBufferStatus(DEVICE_MAC_NW_INTERFACE(nDeviceId,nInterfaceId)->pstruAccessBuffer))
			{
				pstruPacket = fn_NetSim_Packet_GetPacketFromBuffer(DEVICE_MAC_NW_INTERFACE(nDeviceId,nInterfaceId)->pstruAccessBuffer,0);
			}

			/* Get the packet from Upper layer, */
			if(pstruPacket)
			{
				pstruEventDetails->nPacketId = pstruPacket->nPacketId;
				if(pstruPacket->pstruAppData)
					pstruEventDetails->nApplicationId = pstruPacket->pstruAppData->nApplicationId;

				if(pstruPacket->nControlDataType/100 != MAC_PROTOCOL_IEEE802_15_4)
				{
					if(WSN_MAC(nDeviceId))
					{
						if(WSN_MAC(nGlobalPANCoordinatorId)->nBeaconMode == BEACON_ENABLE && WSN_MAC(nGlobalPANCoordinatorId)->nMacBeaconOrder <= 14)
						{
							if(pstruSuperFrame->nSuperFrameStatus == CAPMODE)
								ZIGBEE_SLOTTED();
							else
							{
								//wait for CAP
							}
						}
						else
						{
							ZIGBEE_UNSLOTTED();
						}
					}
				}
				else
				{
					//Control packet of IEEE 802.15.4
				}
			}
			break;	
		}
	case MAC_IN_EVENT:
		{
			NetSim_PACKET *pstruPacket;

			WSN_MAC(nDeviceId)->nNodeStatus = IDLE;
			pstruPacket = pstruEventDetails->pPacket;
			pstruPacket->pstruMacData->dArrivalTime = pstruEventDetails->dEventTime;
			pstruPacket->pstruMacData->dStartTime = pstruEventDetails->dEventTime;
			pstruPacket->pstruMacData->dEndTime = pstruEventDetails->dEventTime;

			if(WSN_MAC_HEADER(pstruPacket)->nAckRequestFlag)
			{
				NetSim_PACKET *pstruAckPacket;
				IEEE802_15_4_HEADER *pstruHeader;

				if(!ZIGBEE_CHANGERADIOSTATE(nDeviceId, WSN_PHY(nDeviceId)->nRadioState, TRX_ON_BUSY))
				{
					return 0;
				}

				WSN_MAC(nDeviceId)->nNodeStatus = TX_MODE;

				pstruAckPacket = fn_NetSim_Packet_CreatePacket(MAC_LAYER);
				pstruAckPacket->dEventTime = pstruEventDetails->dEventTime;
				pstruAckPacket->nControlDataType = ACK_FRAME;
				pstruAckPacket->nPacketType = PacketType_Control;
				pstruAckPacket->nTransmitterId = pstruPacket->nReceiverId;
				pstruAckPacket->nReceiverId = pstruPacket->nTransmitterId;
				pstruAckPacket->nSourceId = pstruAckPacket->nTransmitterId;
				add_dest_to_packet(pstruAckPacket, pstruAckPacket->nReceiverId);
				pstruAckPacket->pstruMacData->dArrivalTime = pstruAckPacket->dEventTime;
				pstruAckPacket->pstruMacData->dStartTime = pstruAckPacket->dEventTime;
				pstruAckPacket->pstruMacData->dEndTime = pstruAckPacket->dEventTime;
				pstruAckPacket->pstruMacData->szSourceMac = (pstruPacket->pstruMacData->szDestMac);
				pstruAckPacket->pstruMacData->szDestMac = (pstruPacket->pstruMacData->szSourceMac);
				pstruAckPacket->pstruMacData->nMACProtocol = MAC_PROTOCOL_IEEE802_15_4;

				pstruAckPacket->pstruMacData->dOverhead = 5;
				pstruAckPacket->pstruMacData->dPacketSize = pstruAckPacket->pstruMacData->dPayload + pstruPacket->pstruMacData->dOverhead;
				pstruEventDetails->dPacketSize = pstruAckPacket->pstruMacData->dPacketSize;

				pstruHeader = fnpAllocateMemory(1,sizeof(IEEE802_15_4_HEADER));
				pstruAckPacket->pstruMacData->Packet_MACProtocol = pstruHeader;

				pstruEventDetails->pPacket = pstruAckPacket;

				pstruEventDetails->nPacketId = 0;
				pstruEventDetails->dEventTime += 12 * dUnitSymbolTime;
				pstruAckPacket->nPacketId = 0;

				pstruEventDetails->nEventType = PHYSICAL_OUT_EVENT;
				pstruEventDetails->nSubEventType =0;
				fnpAddEvent(pstruEventDetails);
				pstruEventDetails->dEventTime -= 12 * dUnitSymbolTime;
			}
			if(pstruPacket->nControlDataType != ACK_FRAME && pstruPacket->nControlDataType/100 != MAC_PROTOCOL_IEEE802_15_4)
			{
				pstruMetrics[pstruEventDetails->nDeviceId-1]->pstruIEEE802_15_4_Metrics->nPacketReceived++;
			}
			if(pstruPacket->nControlDataType/100 != MAC_PROTOCOL_IEEE802_15_4)
			{
				//Prepare the Network in event details
				pstruPacket->pstruMacData->dOverhead -= 5;
				pstruPacket->pstruMacData->dPacketSize = pstruPacket->pstruMacData->dPayload + pstruPacket->pstruMacData->dOverhead;
				pstruEventDetails->dPacketSize = pstruPacket->pstruMacData->dPacketSize;
				pstruEventDetails->pPacket = pstruPacket;
				pstruEventDetails->nEventType = NETWORK_IN_EVENT;
				pstruEventDetails->nSubEventType = 0;
				pstruEventDetails->nPacketId=pstruPacket->nPacketId;
				if(pstruPacket->pstruAppData)
				{
					pstruEventDetails->nSegmentId=pstruPacket->pstruAppData->nSegmentId;
					pstruEventDetails->nApplicationId=pstruPacket->pstruAppData->nApplicationId;
				}
				fn_NetSim_Zigbee_FreePacket(pstruPacket);
				pstruEventDetails->nProtocolId = fn_NetSim_Stack_GetNWProtocol(pstruEventDetails->nDeviceId);
				//Add Network in event
				fnpAddEvent(pstruEventDetails);
			}
			else if(pstruPacket->nControlDataType == BEACON_FRAME)
			{
				BEACONFRAME *pstruBeaconFrame;
				pstruMetrics[pstruEventDetails->nDeviceId-1]->pstruIEEE802_15_4_Metrics->nBeaconReceived++;
				pstruBeaconFrame = WSN_MAC_HEADER(pstruPacket)->pstruBeaconFrame;
				if(WSN_MAC(pstruEventDetails->nDeviceId)->nBeaconReceivedFlag == 0 && WSN_MAC(pstruEventDetails->nDeviceId)->nLastBeaconId < pstruBeaconFrame->nBeaconId)
				{
					WSN_MAC(pstruEventDetails->nDeviceId)->nLastBeaconId = pstruBeaconFrame->nBeaconId;
					pstruEventDetails->nEventType = PHYSICAL_OUT_EVENT;
					pstruEventDetails->pPacket->nTransmitterId = pstruEventDetails->nDeviceId;
					pstruEventDetails->pPacket->nReceiverId = 0;
					//fnpAddEvent(pstruEventDetails);
					pstruMetrics[pstruEventDetails->nDeviceId-1]->pstruIEEE802_15_4_Metrics->nBeaconForwarded++;
				}
				else
				{
					fn_NetSim_Packet_FreePacket(pstruEventDetails->pPacket);
				}

			}
			else if(pstruPacket->nControlDataType == ACK_FRAME)
			{
#ifdef _NETSIM_WATCHDOG_
				NetSim_PACKET* packet1 = WSN_MAC(pstruEventDetails->nDeviceId)->pstruBuffer;

				UINT dummy = WSN_MAC(pstruEventDetails->nDeviceId)->nRetryCount;
				if (WSN_MAC(pstruEventDetails->nDeviceId)->nRetryCount > 0) //first time only
				{
					add_watchdog_timer(pstruEventDetails->nDeviceId, packet1);
				}
#endif // _NETSIM_WATCHDOG_

				pstruMetrics[pstruEventDetails->nDeviceId-1]->pstruIEEE802_15_4_Metrics->nAckReceived++;

				WSN_MAC(pstruEventDetails->nDeviceId)->nRetryCount = 0;
				if(WSN_MAC(pstruEventDetails->nDeviceId)->pstruBuffer)
				{
					if(WSN_MAC(pstruEventDetails->nDeviceId)->pstruBuffer->ReceiveAckNotification)
					{
						WSN_MAC(pstruEventDetails->nDeviceId)->pstruBuffer->ReceiveAckNotification(pstruPacket);
					}
					fn_NetSim_Packet_FreePacket(WSN_MAC(pstruEventDetails->nDeviceId)->pstruBuffer);
					WSN_MAC(pstruEventDetails->nDeviceId)->pstruBuffer = NULL;
				}

				if(fn_NetSim_GetBufferStatus(NETWORK->ppstruDeviceList[pstruEventDetails->nDeviceId-1]->ppstruInterfaceList[pstruEventDetails->nInterfaceId-1]->pstruAccessInterface->pstruAccessBuffer))
				{
					NetSim_PACKET* packet = fn_NetSim_Packet_GetPacketFromBuffer(NETWORK->ppstruDeviceList[pstruEventDetails->nDeviceId-1]->ppstruInterfaceList[pstruEventDetails->nInterfaceId-1]->pstruAccessInterface->pstruAccessBuffer,0);
					WSN_MAC(pstruEventDetails->nDeviceId)->nNodeStatus = IDLE;
					//Prepare the mac out event details
					pstruEventDetails->dEventTime += 40 * dUnitSymbolTime;
					pstruEventDetails->pPacket = NULL;
					if(packet->pstruAppData)
					{
						pstruEventDetails->nApplicationId = packet->pstruAppData->nApplicationId;
						pstruEventDetails->nSegmentId = packet->pstruAppData->nSegmentId;
					}
					else
					{
						pstruEventDetails->nApplicationId = 0;
						pstruEventDetails->nSegmentId = 0;
					}
					pstruEventDetails->dPacketSize = fnGetPacketSize(packet);
					pstruEventDetails->nPacketId = packet->nPacketId;

					pstruEventDetails->nEventType = MAC_OUT_EVENT;
					pstruEventDetails->nSubEventType = 0;
					pstruEventDetails->nProtocolId = MAC_PROTOCOL_IEEE802_15_4;
					//Add mac out event
					fnpAddEvent(pstruEventDetails);
				}

				fn_NetSim_Packet_FreePacket(pstruPacket);
			}
			break;
		}
	case PHYSICAL_OUT_EVENT:
		{
			NetSim_EVENTDETAILS pevent;
			NETSIM_ID nLoop;
			NetSim_PACKET* pstruPacket;
			NETSIM_ID nLink_Id;					/*Link id b/w Current device and connected device Id*/
			NETSIM_ID nDevice_Id;					/*Device id From Event details*/
			int nDevice_Type;				/*Device Type From Event details*/
			NETSIM_ID nDevice_PortId;			/*Device port id From Event details*/
			double dTxTime=0.0f;			/*Stores Transmission Time*/
			double dDataRate;
			bool txFlag=false;
		
			memcpy(&pevent,pstruEventDetails,sizeof pevent);
			/*Get the current device Type*/
			nDevice_Type=pstruEventDetails->nDeviceType;
			/*Get the current device Id*/
			nDevice_Id=pstruEventDetails->nDeviceId;
			/*Get the current device Port ID*/
			nDevice_PortId=pstruEventDetails->nInterfaceId;

			nLink_Id = DEVICE_PHYLAYER(nDevice_Id,nDevice_PortId)->nLinkId;

			pstruPacket = pstruEventDetails->pPacket;
			pstruPacket->pstruPhyData->dArrivalTime = pstruEventDetails->dEventTime;
			pstruPacket->pstruPhyData->dPayload = pstruPacket->pstruMacData->dPacketSize;
			//6 byte overhead
			if(pstruPacket->nControlDataType == ACK_FRAME)
				pstruPacket->pstruPhyData->dOverhead = 1;
			else
				pstruPacket->pstruPhyData->dOverhead = 6;
			pstruPacket->pstruPhyData->dPacketSize = pstruPacket->pstruPhyData->dPayload + pstruPacket->pstruPhyData->dOverhead;
			pstruEventDetails->dPacketSize = pstruPacket->pstruPhyData->dPacketSize;

			if(NETWORK->ppstruNetSimLinks[nLink_Id-1]->puniMedProp.pstruWirelessLink.dDataRateUp)
				dDataRate = NETWORK->ppstruNetSimLinks[nLink_Id-1]->puniMedProp.pstruWirelessLink.dDataRateUp;
			else
				dDataRate = 0.250;

			dTxTime = pstruPacket->pstruPhyData->dPacketSize*8/dDataRate;

			if(pstruPacket->pstruPhyData->dArrivalTime <= NETWORK->ppstruDeviceList[nDevice_Id-1]->ppstruInterfaceList[nDevice_PortId-1]->pstruPhysicalLayer->dLastPacketEndTime)
			{
				pstruPacket->pstruPhyData->dStartTime = NETWORK->ppstruDeviceList[nDevice_Id-1]->ppstruInterfaceList[nDevice_PortId-1]->pstruPhysicalLayer->dLastPacketEndTime;
			}
			else
				pstruPacket->pstruPhyData->dStartTime = pstruPacket->pstruPhyData->dArrivalTime;

			pstruPacket->pstruPhyData->dEndTime = pstruPacket->pstruPhyData->dStartTime + dTxTime;
			NETWORK->ppstruDeviceList[nDevice_Id-1]->ppstruInterfaceList[nDevice_PortId-1]->pstruPhysicalLayer->dLastPacketEndTime=pstruPacket->pstruPhyData->dEndTime;
#ifdef _NETSIM_WATCHDOG_
			if (pstruEventDetails->pPacket->nPacketType != PacketType_Control)
				add_to_sent_packet(pstruEventDetails->nDeviceId, pstruEventDetails->pPacket);
#endif // _NETSIM_WATCHDOG_

			for(nLoop=1; nLoop<=NETWORK->ppstruNetSimLinks[nLink_Id-1]->puniDevList.pstruMP2MP.nConnectedDeviceCount; nLoop++)
			{
				NETSIM_ID ncon = NETWORK->ppstruNetSimLinks[nLink_Id-1]->puniDevList.pstruMP2MP.anDevIds[nLoop-1];
				if(ncon == pstruPacket->nTransmitterId)
					continue;

				WSN_PHY(ncon)->dTotalReceivedPower += GET_RX_POWER_mw(pstruPacket->nTransmitterId,ncon,pstruEventDetails->dEventTime);

				if(ncon==pstruPacket->nReceiverId || pstruPacket->nReceiverId == 0)
				{
					NetSim_PACKET* packet = pstruPacket;
					if(pstruPacket->nReceiverId == 0)
					{
						packet = fn_NetSim_Packet_CopyPacket(pstruPacket);
					}

					if(WSN_PHY(ncon)->dTotalReceivedPower - GET_RX_POWER_mw(pstruPacket->nTransmitterId,ncon,pstruEventDetails->dEventTime) >= WSN_PHY(ncon)->dReceiverSensivity)
						packet->nPacketStatus = PacketStatus_Collided;

					if(GET_RX_POWER_dbm(pstruPacket->nTransmitterId,ncon,pstruEventDetails->dEventTime) > MW_TO_DBM(WSN_PHY(ncon)->dReceiverSensivity))
					{
						if(WSN_PHY(ncon)->nRadioState == RX_ON_IDLE)
						{

							if (!ZIGBEE_CHANGERADIOSTATE(ncon, WSN_PHY(ncon)->nRadioState, RX_ON_BUSY))
							{
								if (isBroadcastPacket(packet) ||
									isMulticastPacket(packet))
									continue; // Other node maybe alive.
								else
									return 0;
							}
							// Metrics
							WSN_MAC(ncon)->nNodeStatus = RX_MODE;
							txFlag=true;
							pstruEventDetails->dEventTime = packet->pstruPhyData->dEndTime;
							pstruEventDetails->dPacketSize = packet->pstruPhyData->dPacketSize;
							pstruEventDetails->nDeviceId = ncon;
							packet->nReceiverId = pstruEventDetails->nDeviceId;
							pstruEventDetails->nDeviceType = NETWORK->ppstruDeviceList[ncon-1]->nDeviceType;
							pstruEventDetails->nInterfaceId = fn_NetSim_Stack_GetWirelessInterface(nLink_Id,ncon);
							pstruEventDetails->pPacket = packet;
							pstruEventDetails->nEventType = PHYSICAL_IN_EVENT;
							fnpAddEvent(pstruEventDetails);
						}
					}
				}
			}
			if(WSN_MAC_HEADER(pstruPacket)->nAckRequestFlag)
			{
				pevent.nEventType = TIMER_EVENT;
				pevent.nSubEventType = ACK_TIMEOUT;
				pevent.dEventTime = pstruPacket->pstruPhyData->dEndTime + WSN_PHY(nDevice_Id)->macAckWaitDuration*4/dDataRate;
				pevent.pPacket = NULL;
				fnpAddEvent(&pevent);
			}


			if(pstruPacket->nControlDataType != ACK_FRAME && pstruPacket->nControlDataType/100 != MAC_PROTOCOL_IEEE802_15_4 && txFlag == true)
			{
				pstruMetrics[nDevice_Id-1]->pstruIEEE802_15_4_Metrics->nPacketTransmitted++;
			}
			if(pstruPacket->nControlDataType == ACK_FRAME && txFlag == true)
			{
				pstruMetrics[nDevice_Id-1]->pstruIEEE802_15_4_Metrics->nAckTransmitted++;
			}

			//store the end time
			NETWORK->ppstruDeviceList[nDevice_Id-1]->ppstruInterfaceList[nDevice_PortId-1]->pstruPhysicalLayer->dLastPacketEndTime = pstruPacket->pstruPhyData->dEndTime;
			if(!pstruPacket->nReceiverId || txFlag==false) 
				fn_NetSim_Packet_FreePacket(pstruPacket);
			//Add timer event to update the medium
			pstruEventDetails->dEventTime=NETWORK->ppstruDeviceList[nDevice_Id-1]->ppstruInterfaceList[nDevice_PortId-1]->pstruPhysicalLayer->dLastPacketEndTime+1;
			pstruEventDetails->nDeviceId = nDevice_Id;
			pstruEventDetails->nEventType = TIMER_EVENT;
			pstruEventDetails->nSubEventType = UPDATE_MEDIUM;
			pstruEventDetails->pPacket = NULL;
			//Add the physical in event
			fnpAddEvent(pstruEventDetails);
			break;
		}
	case PHYSICAL_IN_EVENT: 
		{
			NetSim_PACKET *pstruPacket;
			PACKET_STATUS nPacketStatus;
			double dFadingPower = 0;
			NETSIM_ID ifid;
			double SNR;
			double dBER;

			pstruPacket = pstruEventDetails->pPacket;
			if(pstruPacket->nReceiverId && pstruPacket->nReceiverId != pstruEventDetails->nDeviceId)
			{
				fnNetSimError("Different device packet received..");
				assert(false);
				return 0;
			}


			if(!ZIGBEE_CHANGERADIOSTATE(pstruEventDetails->nDeviceId, WSN_PHY(pstruEventDetails->nDeviceId)->nRadioState, RX_ON_IDLE))
				return 0;

			if(WSN_PHY(pstruEventDetails->nDeviceId)->dTotalReceivedPower - GET_RX_POWER_mw(pstruPacket->nTransmitterId,pstruPacket->nReceiverId,pstruEventDetails->dEventTime) >= WSN_PHY(pstruEventDetails->nDeviceId)->dReceiverSensivity)
				pstruPacket->nPacketStatus = PacketStatus_Collided;


			nPacketStatus = pstruPacket->nPacketStatus;

			double pdbm = GET_RX_POWER_dbm(pstruPacket->nTransmitterId, 
										   pstruPacket->nReceiverId, 
										   pstruEventDetails->dEventTime); 

			//Getting interface id of transmitter
			ifid = fn_NetSim_Stack_GetConnectedInterface(pstruEventDetails->nDeviceId,
				pstruEventDetails->nInterfaceId,
				pstruPacket->nTransmitterId);

			//calculating fading loss
			dFadingPower = _propagation_calculate_fadingloss(
							find_propagation_info(pstruPacket->nTransmitterId, ifid, 
							pstruEventDetails->nDeviceId, pstruEventDetails->nInterfaceId));

			pdbm -= dFadingPower;
			ZIGBEE_SINR(&SNR, WSN_PHY(pstruEventDetails->nDeviceId)->dTotalReceivedPower, DBM_TO_MW(pdbm));

			dBER = fn_NetSim_Zigbee_CalculateBER(SNR);

			if(fn_NetSim_Packet_DecideError(dBER,pstruEventDetails->dPacketSize))
			{
				pstruPacket->nPacketStatus = PacketStatus_Error;
				nPacketStatus = PacketStatus_Error;
			}

			fn_NetSim_WritePacketTrace(pstruPacket);

			if(pstruPacket->nControlDataType == ACK_FRAME)
				pstruPacket->pstruPhyData->dOverhead -= 1;
			else
				pstruPacket->pstruPhyData->dOverhead -= 6;

			pstruPacket->pstruPhyData->dPacketSize = pstruPacket->pstruPhyData->dPayload + pstruPacket->pstruPhyData->dOverhead;
			pstruEventDetails->dPacketSize = pstruPacket->pstruPhyData->dPacketSize;

			if((nPacketStatus == PacketStatus_Error) || (nPacketStatus == PacketStatus_Collided))
			{
				fn_NetSim_Metrics_Add(pstruPacket);
				fn_NetSim_Packet_FreePacket(pstruPacket);
				WSN_MAC(pstruEventDetails->nDeviceId)->nNodeStatus = IDLE;
			}
			else
			{
				pstruPacket->pstruPhyData->dArrivalTime = pstruEventDetails->dEventTime;
				pstruPacket->pstruPhyData->dStartTime = pstruEventDetails->dEventTime;
				pstruPacket->pstruPhyData->dEndTime = pstruEventDetails->dEventTime;

				fn_NetSim_Metrics_Add(pstruPacket);


				//Prepare the mac in event details
				pstruEventDetails->nEventType = MAC_IN_EVENT;
				//Add mac in event
				fnpAddEvent(pstruEventDetails);
			}
			break;
		}
	case TIMER_EVENT:
		{
			switch(nSub_Event_Type)
			{
			case CARRIERSENSE_START:
				{
					if(WSN_MAC(nGlobalPANCoordinatorId)->nBeaconMode == BEACON_ENABLE && WSN_MAC(nGlobalPANCoordinatorId)->nMacBeaconOrder <= 14)//&& pstruSuperFrame->nSuperFrameStatus == CAPMODE)
					{
						if(pstruSuperFrame->nSuperFrameStatus == CAPMODE)
						{
							ZIGBEE_SLOTTED();
						}
						else
						{
							//wait for CAP
						}
					}
					else
					{
						ZIGBEE_UNSLOTTED();
					}
					break;
				}
			case CARRIERSENSE_END:
				{
					if(WSN_MAC(nGlobalPANCoordinatorId)->nBeaconMode == BEACON_ENABLE && WSN_MAC(nGlobalPANCoordinatorId)->nMacBeaconOrder <= 14)//&& pstruSuperFrame->nSuperFrameStatus == CAPMODE)
					{
						if(pstruSuperFrame->nSuperFrameStatus == CAPMODE)
						{
							ZIGBEE_SLOTTED();
						}
						else
						{
							//wait for CAP
						}
					}
					else
					{
						ZIGBEE_UNSLOTTED();
					}
					break;
				}
			case UPDATE_MEDIUM:
				{
					double dtime=pstruEventDetails->dEventTime;
					NETSIM_ID nLink_Id, nConnectionID, nConnectionPortID, nLoop;
					NETSIM_ID nTransmitterID;

					nTransmitterID = pstruEventDetails->nDeviceId;

					ZIGBEE_CHANGERADIOSTATE(nTransmitterID, WSN_PHY(nTransmitterID)->nRadioState, RX_ON_IDLE);
					if(WSN_PHY(nTransmitterID)->nRadioState != RX_OFF)
						WSN_MAC(nTransmitterID)->nNodeStatus = IDLE;
					nLink_Id = fn_NetSim_Stack_GetConnectedDevice(pstruEventDetails->nDeviceId,pstruEventDetails->nInterfaceId,&nConnectionID,&nConnectionPortID);

					for(nLoop=1; nLoop<=NETWORK->ppstruNetSimLinks[nLink_Id-1]->puniDevList.pstruMP2MP.nConnectedDeviceCount; nLoop++)
					{
						NETSIM_ID ncon = NETWORK->ppstruNetSimLinks[nLink_Id-1]->puniDevList.pstruMP2MP.anDevIds[nLoop-1];
						if(ncon != pstruEventDetails->nDeviceId)
						{
							WSN_PHY(ncon)->dTotalReceivedPower -= GET_RX_POWER_mw(nTransmitterID,ncon,pstruEventDetails->dEventTime);

							if(WSN_PHY(ncon)->dTotalReceivedPower < WSN_PHY(ncon)->dReceiverSensivity)
								WSN_PHY(ncon)->dTotalReceivedPower = 0;
						}
					}

					for(nLoop=1; nLoop<=NETWORK->ppstruNetSimLinks[nLink_Id-1]->puniDevList.pstruMP2MP.nConnectedDeviceCount; nLoop++)
					{
						nDeviceId= NETWORK->ppstruNetSimLinks[nLink_Id-1]->puniDevList.pstruMP2MP.anDevIds[nLoop-1];
						IEEE802_15_4_MAC_VAR* macVar=WSN_MAC(nDeviceId);
						IEEE802_15_4_PHY_VAR* phyVar=WSN_PHY(nDeviceId);
						if((phyVar->dTotalReceivedPower < phyVar->dReceiverSensivity) &&
							(macVar->nNodeStatus == IDLE) && (phyVar->nRadioState == RX_ON_IDLE) &&
							fn_NetSim_GetBufferStatus(NETWORK->ppstruDeviceList[nDeviceId-1]->ppstruInterfaceList[0]->pstruAccessInterface->pstruAccessBuffer))
						{
							NetSim_PACKET* packet = fn_NetSim_Packet_GetPacketFromBuffer(NETWORK->ppstruDeviceList[nDeviceId-1]->ppstruInterfaceList[0]->pstruAccessInterface->pstruAccessBuffer,0);

							//Add Mac out event
							if(nDeviceId==nTransmitterID)
								pstruEventDetails->dEventTime += 40*dUnitSymbolTime;
							else
								pstruEventDetails->dEventTime=dtime;
							pstruEventDetails->nDeviceId = nDeviceId;
							pstruEventDetails->nEventType = MAC_OUT_EVENT;
							pstruEventDetails->nSubEventType = 0;
							pstruEventDetails->pPacket = NULL;
							pstruEventDetails->dPacketSize = fnGetPacketSize(packet);
							pstruEventDetails->nPacketId =packet->nPacketId;
							if(packet->pstruAppData)
							{
								pstruEventDetails->nApplicationId = packet->pstruAppData->nApplicationId;
								pstruEventDetails->nSegmentId = packet->pstruAppData->nSegmentId;
							}
							else
							{
								pstruEventDetails->nApplicationId=0;
								pstruEventDetails->nSegmentId=0;
							}
							fnpAddEvent(pstruEventDetails);
						}
					}
					break;
				}
			case ACK_TIMEOUT:
				{
					if(WSN_MAC(nDeviceId)->pstruBuffer)
					{
						WSN_MAC(nDeviceId)->nRetryCount++;
						if(WSN_MAC(nDeviceId)->nRetryCount <= WSN_MAC(nDeviceId)->nMacMaxFrameRetries)
						{
							pstruEventDetails->nEventType = MAC_OUT_EVENT;
							pstruEventDetails->nSubEventType = 0;
							fnpAddEvent(pstruEventDetails);
						}
						else
						{
							WSN_MAC(nDeviceId)->nRetryCount = 0;
							if(WSN_MAC(nDeviceId)->pstruBuffer->DropNotification)
								WSN_MAC(nDeviceId)->pstruBuffer->DropNotification(WSN_MAC(nDeviceId)->pstruBuffer);
							//Drop the packet
							WSN_MAC(nDeviceId)->pstruBuffer->nPacketStatus = PacketStatus_Buffer_Dropped;
							fn_NetSim_WritePacketTrace(WSN_MAC(nDeviceId)->pstruBuffer);
							fn_NetSim_Packet_FreePacket(WSN_MAC(nDeviceId)->pstruBuffer);
							WSN_MAC(nDeviceId)->pstruBuffer = NULL;
							//try next packet
							if(WSN_MAC(nDeviceId)->nNodeStatus==IDLE)
								if(fn_NetSim_GetBufferStatus(NETWORK->ppstruDeviceList[pstruEventDetails->nDeviceId-1]->ppstruInterfaceList[pstruEventDetails->nInterfaceId-1]->pstruAccessInterface->pstruAccessBuffer))
								{
									NetSim_PACKET* packet = fn_NetSim_Packet_GetPacketFromBuffer(NETWORK->ppstruDeviceList[pstruEventDetails->nDeviceId-1]->ppstruInterfaceList[pstruEventDetails->nInterfaceId-1]->pstruAccessInterface->pstruAccessBuffer,0);
									//Prepare the mac out event details
									pstruEventDetails->dEventTime += (double)40 * dUnitSymbolTime;
									pstruEventDetails->pPacket = NULL;
									if(packet->pstruAppData)
									{
										pstruEventDetails->nApplicationId = packet->pstruAppData->nApplicationId;
										pstruEventDetails->nSegmentId = packet->pstruAppData->nSegmentId;
									}
									else
									{
										pstruEventDetails->nApplicationId = 0;
										pstruEventDetails->nSegmentId = 0;
									}
									pstruEventDetails->dPacketSize = fnGetPacketSize(packet);
									pstruEventDetails->nPacketId = packet->nPacketId;

									pstruEventDetails->nEventType = MAC_OUT_EVENT;
									pstruEventDetails->nSubEventType = 0;
									pstruEventDetails->nProtocolId = MAC_PROTOCOL_IEEE802_15_4;
									//Add mac out event
									fnpAddEvent(pstruEventDetails);
								}
						}
					}
					break;
				}
			case SUPERFRAME_EVENT:
				{
					ZIGBEE_SUPERFRAME();
				}
				break;
			case BEACON_TRANSMISSION_END:
				{
					NETSIM_ID nLoop;
					//For the data transmission
					for(nLoop=0;nLoop<NETWORK->nDeviceCount;nLoop++)
					{
						if(fn_NetSim_GetBufferStatus(DEVICE_INTERFACE(nLoop+1,1)->pstruAccessInterface->pstruAccessBuffer))
						{
							WSN_MAC(nLoop+1)->nNodeStatus = IDLE;
							WSN_MAC(nLoop+1)->nRetryCount = 0;
							WSN_MAC(nLoop+1)->nNoOfBackOff = 0;

							pstruEventDetails->nDeviceId = (NETSIM_ID)nLoop+1;
							pstruEventDetails->nInterfaceId = 1;
							pstruEventDetails->pPacket = fn_NetSim_Packet_GetPacketFromBuffer(DEVICE_MAC_NW_INTERFACE(pstruEventDetails->nDeviceId,pstruEventDetails->nInterfaceId)->pstruAccessBuffer,1);
							pstruEventDetails->dPacketSize = pstruEventDetails->pPacket->pstruNetworkData->dPacketSize;
							pstruEventDetails->nEventType = MAC_OUT_EVENT;
							pstruEventDetails->nSubEventType = 0;
							pstruEventDetails->nProtocolId = MAC_PROTOCOL_IEEE802_15_4;
							fnpAddEvent(pstruEventDetails);
						}
					}
					//Update the super frame status
					pstruSuperFrame->nSuperFrameStatus = CAPMODE;
					pstruEventDetails->dEventTime = pstruEventDetails->dEventTime + pstruSuperFrame->dCAPLength;
					pstruEventDetails->nDeviceId = nGlobalPANCoordinatorId;
					pstruEventDetails->nInterfaceId = 1;
					pstruEventDetails->nEventType = TIMER_EVENT;
					pstruEventDetails->nSubEventType = CAP_END;
					pstruEventDetails->nProtocolId = MAC_PROTOCOL_IEEE802_15_4;
					fnpAddEvent(pstruEventDetails);


					break;
				}
			case CAP_END:
				{
					pstruSuperFrame->nSuperFrameStatus = CFPMODE;
					pstruEventDetails->dEventTime = pstruEventDetails->dEventTime + pstruSuperFrame->dCFPLength;
					pstruEventDetails->nDeviceId = nGlobalPANCoordinatorId;
					pstruEventDetails->nInterfaceId = 1;
					pstruEventDetails->nEventType = TIMER_EVENT;
					pstruEventDetails->nSubEventType = CFP_END;
					pstruEventDetails->nProtocolId = MAC_PROTOCOL_IEEE802_15_4;
					fnpAddEvent(pstruEventDetails);
				}
				break;
			case CFP_END:
				{
					NETSIM_ID nLink_Id, nConnectionID, nConnectionPortID,nLoop;
					nLink_Id = fn_NetSim_Stack_GetConnectedDevice(pstruEventDetails->nDeviceId,pstruEventDetails->nInterfaceId,&nConnectionID,&nConnectionPortID);
					for(nLoop=1; nLoop<=NETWORK->ppstruNetSimLinks[nLink_Id-1]->puniDevList.pstruMP2MP.nConnectedDeviceCount; nLoop++)
					{
						nDeviceId = NETWORK->ppstruNetSimLinks[nLink_Id-1]->puniDevList.pstruMP2MP.anDevIds[nLoop-1];
						WSN_MAC(nDeviceId)->nNodeStatus = OFF;
						ZIGBEE_CHANGERADIOSTATE(nDeviceId, WSN_PHY(nDeviceId)->nRadioState, SLEEP);

					}
					//Set the super frame status as inactive
					pstruSuperFrame->nSuperFrameStatus = INACTIVEMODE;
				}
				break;

			case SUBEVENT_GETLINKQUALITY:
			{
				double* pass = pstruEventDetails->szOtherDetails;
				NETSIM_ID trx = (NETSIM_ID)pass[0];
				NETSIM_ID tri = (NETSIM_ID)pass[1];
				NETSIM_ID rx = (NETSIM_ID)pass[2];
				NETSIM_ID ri = (NETSIM_ID)pass[3];
				pass[4] = zigbee_get_link_quality(trx,tri,rx,ri);
			}
			break;
#ifdef _NETSIM_WATCHDOG_
			case WATCHDOG_TIMER:
				watchdog_timer_execute();
				break;
#endif // _NETSIM_WATCHDOG_

			default:
				fnNetSimError("Invalid event %d for zigbee protocol\n", pstruEventDetails->nSubEventType);
				break;
			}
			break;
		}
	default:
		fnNetSimError("Invalid event 5d for zigbee protocol\n",pstruEventDetails->nEventType);
		break;
	}
	return 0;
}
/** This function is called by NetworkStack.dll to configure the Zigbee Protocol related devices. */
_declspec(dllexport) int fn_NetSim_Zigbee_Configure(void** var)
{
	return fn_NetSim_Zigbee_Configure_F(var);
}
/**  This function Initializes the Zigbee protocol parameters.*/
_declspec (dllexport) int fn_NetSim_Zigbee_Init(struct stru_NetSim_Network* NETWORK_Formal, \
	NetSim_EVENTDETAILS* pstruEventDetails_Formal, char* pszAppPath_Formal, \
	char* pszWritePath_Formal, int nVersion_Type, void** fnPointer)
{
	pstruEventDetails = pstruEventDetails_Formal;
	NETWORK = NETWORK_Formal;
	pszAppPath = pszAppPath_Formal;
	pszIOPath = pszWritePath_Formal;
	fn_NetSim_Zigbee_Init_F(NETWORK_Formal, pstruEventDetails_Formal, pszAppPath_Formal, \
		pszWritePath_Formal, nVersion_Type, fnPointer);
#ifdef _NETSIM_WATCHDOG_
	watchdog_init();
#endif
	return 0;
}
/** 
This function is called by NetworkStack.dll, once the simulation end, this function free
the allocated memory for the network.	 
*/
_declspec(dllexport) int fn_NetSim_Zigbee_Finish()
{
#ifdef _NETSIM_WATCHDOG_
	watchdog_close();
#endif // _NETSIM_WATCHDOG_

	fn_NetSim_Zigbee_Finish_F();
	return 0;
}	

_declspec (dllexport) char *fn_NetSim_Zigbee_Trace(int nSubEvent)
{
#ifdef _NETSIM_WATCHDOG_
	switch (nSubEvent)
	{
	case WATCHDOG_TIMER:
		return "WATCHDOG_TIMER";
	}
#endif // _NETSIM_WATCHDOG_

	return (fn_NetSim_Zigbee_Trace_F(nSubEvent));
}
/** This function is called by NetworkStack.dll,to free the packet. */
_declspec(dllexport) int fn_NetSim_Zigbee_FreePacket(NetSim_PACKET* pstruPacket)
{
	return fn_NetSim_Zigbee_FreePacket_F(pstruPacket);	
}
/** This function is called by NetworkStack.dll, to copy the Zigbee destination packet to source packet. */
_declspec(dllexport) int fn_NetSim_Zigbee_CopyPacket(NetSim_PACKET* pstruSrcPacket,NetSim_PACKET* pstruDestPacket)
{
	return fn_NetSim_Zigbee_CopyPacket_F(pstruSrcPacket,pstruDestPacket);	
}
/** This function write the metrics in metrics.txt */
_declspec(dllexport) int fn_NetSim_Zigbee_Metrics(PMETRICSWRITER metricsWriter)
{
	return fn_NetSim_Zigbee_Metrics_F(metricsWriter);	
}

/**
This function is to configure the Zigbee protocol packet trace parameter. 
This function return a string which has the parameters separated by comma.
*/
_declspec(dllexport) char* fn_NetSim_Zigbee_ConfigPacketTrace(const void* xmlNetSimNode)
{
	xmlNetSimNode;
	return "";
}

/**
This function is called while writing the Packet trace for Zigbee protocol. 
This function is called for every packet while writing the packet trace.
*/
_declspec(dllexport) int fn_NetSim_Zigbee_WritePacketTrace(NetSim_PACKET* pstruPacket,char** ppszTrace)
{
	ppszTrace;
	pstruPacket;
	return 1;
}
/** This function is used to form the channel as well as to allocate the frequency,channel number */
int fn_NetSim_Zigbee_ChannelFormation(CHANNELS** pstruChannelList)
{
	*pstruChannelList = calloc(1,sizeof(CHANNELS));
	(*pstruChannelList)->nChannelStatus = IDLE;
	(*pstruChannelList)->dFrequency = WSN_PHY(nGlobalPANCoordinatorId)->dFrequencyBand_MHz;
	(*pstruChannelList)->nChannelNumber = (int)((*pstruChannelList)->dFrequency - 2405)/5 +11;
	return 1;
}

_declspec(dllexport) int fn_NetSim_ZigBee_SinkNodeApp()
{
	fn_NetSim_Packet_FreePacket(pstruEventDetails->pPacket);
	return 0;
}

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
#include "802_15_4.h"
/** 
If periodic beacons are not being used in the PAN or if a beacon could not be located in a beacon-enabled PAN, 
the MAC sublayer shall transmit using the unslotted version of the CSMA-CA algorithm. 

Nonbeacon-enabled PANs use an unslotted CSMA-CA channel access mechanism. Each time a device wishes to transmit data 
frames or MAC commands, it waits for a random period. If the channel is found to be idle, following the random backoff, 
the device transmits its data. If the channel is found to be busy following the random backoff, the device waits for 
another random period before trying to access the channel again
*/
int fn_NetSim_Zigbee_UnslottedCSMACA()
{
	int nReturn;
	double dTime;
	NETSIM_ID nDeviceID;
	IEEE802_15_4_MAC_VAR* macVar=WSN_MAC(pstruEventDetails->nDeviceId);

	nDeviceID = pstruEventDetails->nDeviceId;
	if(WSN_PHY(nDeviceID)->nRadioState != RX_ON_IDLE)
	{
		macVar->nNodeStatus = IDLE;
		return 0;
	}

	switch(pstruEventDetails->nSubEventType)
	{
	case CARRIERSENSE_START:
		{
			if(macVar->nNodeStatus != BACKOFF_MODE)
				return 0;//other mode.

			nReturn = ZIGBEE_CCA(WSN_PHY(nDeviceID)->dTotalReceivedPower, WSN_PHY(nDeviceID)->enumCCAMode,
				WSN_PHY(nDeviceID)->dReceiverSensivity, WSN_PHY(nDeviceID)->dEDThreshold);

			if(nReturn == CHANNEL_IDLE)
			{
				macVar->nNodeStatus = CCA_MODE;
				pstruEventDetails->dEventTime += 8 * dUnitSymbolTime;
				pstruEventDetails->nEventType = TIMER_EVENT;
				pstruEventDetails->nSubEventType = CARRIERSENSE_END;
				pstruEventDetails->nProtocolId = MAC_PROTOCOL_IEEE802_15_4;
				fnpAddEvent(pstruEventDetails);	
				return 1;
			}
			else
			{
				goto CARRIERSENSEFAILED;
			}
			break;
		}
	case CARRIERSENSE_END:
		{
			if(WSN_MAC(nDeviceID)->nNodeStatus != CCA_MODE)
				return 0;//other mode.
			nReturn = ZIGBEE_CCA(WSN_PHY(nDeviceID)->dTotalReceivedPower, WSN_PHY(nDeviceID)->enumCCAMode,
				WSN_PHY(nDeviceID)->dReceiverSensivity, WSN_PHY(nDeviceID)->dEDThreshold);

			if(nReturn == CHANNEL_IDLE)
			{
				IEEE802_15_4_HEADER *pstruHeader;

				if(WSN_MAC(nDeviceID)->pstruBuffer)
				{
					pstruEventDetails->pPacket = WSN_MAC(nDeviceID)->pstruBuffer;
				}
				else if(fn_NetSim_GetBufferStatus(DEVICE_MAC_NW_INTERFACE(nDeviceID,pstruEventDetails->nInterfaceId)->pstruAccessBuffer))
				{
					pstruEventDetails->pPacket = fn_NetSim_Packet_GetPacketFromBuffer(DEVICE_MAC_NW_INTERFACE(nDeviceID,pstruEventDetails->nInterfaceId)->pstruAccessBuffer,1);
				}
				else
					assert(false);

				//Append mac details in packet
				pstruEventDetails->pPacket->pstruMacData->dArrivalTime = pstruEventDetails->pPacket->pstruNetworkData->dEndTime;
				pstruEventDetails->pPacket->pstruMacData->dStartTime = pstruEventDetails->dEventTime;
				pstruEventDetails->pPacket->pstruMacData->dEndTime = pstruEventDetails->dEventTime;
				//5 byte overhead
				pstruEventDetails->pPacket->pstruMacData->dOverhead = 5;
				pstruEventDetails->pPacket->pstruMacData->dPayload = pstruEventDetails->pPacket->pstruNetworkData->dPacketSize;
				pstruEventDetails->pPacket->pstruMacData->dPacketSize = pstruEventDetails->pPacket->pstruMacData->dOverhead + \
					pstruEventDetails->pPacket->pstruMacData->dPayload;
				pstruEventDetails->dPacketSize = pstruEventDetails->pPacket->pstruMacData->dPacketSize;
				pstruEventDetails->pPacket->pstruMacData->nMACProtocol = MAC_PROTOCOL_IEEE802_15_4;

				//IEEE 802.15.4 header
				if(!pstruEventDetails->pPacket->pstruMacData->Packet_MACProtocol)
				{
					pstruHeader = fnpAllocateMemory(1,sizeof(IEEE802_15_4_HEADER));
					pstruEventDetails->pPacket->pstruMacData->Packet_MACProtocol = pstruHeader;
				}

				
				if(WSN_MAC(nDeviceID)->nAckRequestFlag == ENABLE && pstruEventDetails->pPacket->nReceiverId)
				{
					WSN_MAC_HEADER(pstruEventDetails->pPacket)->nAckRequestFlag = ENABLE;
					WSN_MAC(nDeviceID)->pstruBuffer = fn_NetSim_Packet_CopyPacket(pstruEventDetails->pPacket);
				}

				if(WSN_PHY(nDeviceID)->nRadioState == RX_ON_IDLE && ZIGBEE_CHANGERADIOSTATE(nDeviceID, WSN_PHY(nDeviceID)->nRadioState, TRX_ON_BUSY))
				{
					//Proceed
					WSN_MAC(nDeviceID)->nNodeStatus = TX_MODE;
				}
				else
				{
					return 0;
				}
				//Write physical out event.
				pstruEventDetails->dEventTime += 12 * dUnitSymbolTime;
				pstruEventDetails->nEventType = PHYSICAL_OUT_EVENT;
				pstruEventDetails->nSubEventType = 0;

				if(pstruEventDetails->pPacket->pstruAppData)
					pstruEventDetails->nSegmentId = pstruEventDetails->pPacket->pstruAppData->nSegmentId;
				//Add physical out event
				fnpAddEvent(pstruEventDetails);
			}
			else
			{
CARRIERSENSEFAILED:
				//check for the receiver
				if(WSN_PHY(nDeviceID)->nRadioState != RX_ON_IDLE)
				{
					return 0;//transmission is in progress
				}
				//Increment Number of backoffs by 1
				WSN_MAC(nDeviceID)->nNoOfBackOff++;
				//Set the Backoff exponent
				if(WSN_MAC(nDeviceID)->nBackoffExponent+1 > WSN_MAC(nDeviceID)->nMacMaxBE)
				{
					WSN_MAC(nDeviceID)->nBackoffExponent = WSN_MAC(nDeviceID)->nMacMaxBE;
				}
				else
					WSN_MAC(nDeviceID)->nBackoffExponent++;
				//Reset the contension window
				WSN_MAC(nDeviceID)->nContentionWindow = 2;
				//Check with max backoff time
				if(WSN_MAC(nDeviceID)->nNoOfBackOff > WSN_MAC(nDeviceID)->nMacMaxCSMABackoff)
				{
					NetSim_PACKET* pstruTempPacket;
					pstruTempPacket = fn_NetSim_Packet_GetPacketFromBuffer(DEVICE_INTERFACE(nDeviceID,1)->pstruAccessInterface->pstruAccessBuffer,1);
					fn_NetSim_Packet_FreePacket(pstruTempPacket);
					/* Go for random backoff then send next packet */
					if(fn_NetSim_GetBufferStatus(DEVICE_INTERFACE(nDeviceID,1)->pstruAccessInterface->pstruAccessBuffer))
					{
						WSN_MAC(nDeviceID)->nNoOfBackOff = 0;
						WSN_MAC(nDeviceID)->nBackoffExponent = WSN_MAC(nDeviceID)->nMacMinBE;
						goto DELAYBACKOFF;						
					}
					else
					{
						WSN_MAC(nDeviceID)->nNodeStatus = IDLE;
						//wait for packet from upper layer
						return 0;
					}
				}
				else
				{
					/* Go for random backoff then retry. */
					goto DELAYBACKOFF;
				}

			}
			break;
		}
	default:
		{
			if(WSN_MAC(nDeviceID)->nNodeStatus == IDLE && WSN_PHY(nDeviceID)->nRadioState == RX_ON_IDLE)
			{
				WSN_MAC(nDeviceID)->nNodeStatus = BACKOFF_MODE;
			}
			else
				return 0;
			WSN_MAC(nDeviceID)->nNoOfBackOff = 0;
			WSN_MAC(nDeviceID)->nBackoffExponent = WSN_MAC(nDeviceID)->nMacMinBE;
DELAYBACKOFF:
			WSN_MAC(nDeviceID)->nNodeStatus = BACKOFF_MODE;
			ZIGBEE_BACKOFF(WSN_MAC(nDeviceID)->nBackoffExponent,
				&dTime,
				WSN_MAC(nDeviceID)->aUnitBackoffPeriod);
			pstruEventDetails->dEventTime += dTime;
			pstruEventDetails->nEventType = TIMER_EVENT;
			pstruEventDetails->nSubEventType = CARRIERSENSE_START;
			pstruEventDetails->nProtocolId = MAC_PROTOCOL_IEEE802_15_4;
			fnpAddEvent(pstruEventDetails);	
			return 1;
		}
	}
	return 1;
}

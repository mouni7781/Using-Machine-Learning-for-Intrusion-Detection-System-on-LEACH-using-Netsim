/************************************************************************************
* Copyright (C) 2015                                                               *
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
/****************************************************
         This file contains code for Intruder detection system. 

Code Flow – 
If _NETSIM_WATCHDOG_ is defined, watchdog timer starts.
Once a packet is forwarded to next hop node, the current node 
Checks for watchdog timer duration if the packet is getting
Forwarded further to destination node or not.

The packet is sent until the watchdog timer expires and
The failure threshold is reached.

****************************************************/

#include "main.h"
#include "802_15_4.h"
#include "watchdog.h"

PWATCHDOG* watchdog;

void watchdog_init()
{
	NETSIM_ID i;
	watchdog = (PWATCHDOG*)calloc(NETWORK->nDeviceCount+1,sizeof* watchdog);
	for(i=1;i<=NETWORK->nDeviceCount;i++)
	{
		watchdog[i] = (PWATCHDOG)calloc(1,sizeof* watchdog[i]);
		watchdog[i]->failureCounter = (unsigned int*)calloc(NETWORK->nDeviceCount+1,sizeof* watchdog[i]->failureCounter);
	}
}

void watchdog_close()
{
	NETSIM_ID i;
	for(i=1;i<=NETWORK->nDeviceCount;i++)
	{
		free(watchdog[i]->failureCounter);
		free(watchdog[i]);
	}
	free(watchdog);
}

void add_to_sent_packet(NETSIM_ID dev,NetSim_PACKET* packet)
{
	PWATCHDOG temp=watchdog[dev];
	NetSim_PACKET* p=temp->sent_packet;
	NETSIM_ID d = get_first_dest_from_packet(packet);

	if(packet->nSourceId == dev || d==dev || packet->nControlDataType%100 == MAC_PROTOCOL_IEEE802_15_4)
		return;
	if(!p)
		temp->sent_packet=fn_NetSim_Packet_CopyPacket(packet);
	else
	{
		while(p->pstruNextPacket)
			p=p->pstruNextPacket;
		p->pstruNextPacket=fn_NetSim_Packet_CopyPacket(packet);
	}
}

void add_watchdog_timer(NETSIM_ID dev,NetSim_PACKET* packet)
{
	NetSim_EVENTDETAILS pevent;
	NETSIM_ID d = get_first_dest_from_packet(packet);

	if(d == packet->nReceiverId)
		return;
	memset(&pevent,0,sizeof pevent);
	pevent.dEventTime=pstruEventDetails->dEventTime+WATCHDOG_TIME;
	pevent.nDeviceId=dev;
	pevent.nDeviceType=DEVICE_TYPE(dev);
	pevent.nEventType=TIMER_EVENT;
	pevent.nInterfaceId=pstruEventDetails->nInterfaceId;
	pevent.nPacketId=0;
	pevent.nProtocolId=MAC_PROTOCOL_IEEE802_15_4;
	pevent.nSubEventType=WATCHDOG_TIMER;
	pevent.pPacket = fn_NetSim_Packet_CopyPacket(packet);
	fnpAddEvent(&pevent);
}

bool check_in_sent_list(NETSIM_ID dev,NetSim_PACKET* packet)
{
	NETSIM_ID rcv=packet->nReceiverId;
	PWATCHDOG rdog=watchdog[rcv];
	NetSim_PACKET* temp = rdog->sent_packet;
	NetSim_PACKET* prev=NULL;

	
	while(temp)
	{
		NETSIM_ID nDestinationId = get_first_dest_from_packet(packet);
		NETSIM_ID tDestinationId = get_first_dest_from_packet(temp);
		if(temp->nSourceId == packet->nSourceId &&
			tDestinationId == nDestinationId &&
			temp->nPacketId == packet->nPacketId &&
			temp->nControlDataType == packet->nControlDataType)
		{
			//matched
			if(!prev)
				rdog->sent_packet=temp->pstruNextPacket;
			else
				prev->pstruNextPacket=temp->pstruNextPacket;
			fn_NetSim_Packet_FreePacket(temp);
			return true;
		}
		prev=temp;
		temp=temp->pstruNextPacket;
	}
	return false;
}

void watchdog_timer_execute()
{
	NetSim_PACKET* packet = pstruEventDetails->pPacket;
	NETSIM_ID dev = pstruEventDetails->nDeviceId;
	NETSIM_ID rcv = packet->nReceiverId;
	if(!check_in_sent_list(dev,packet))
	{
		PWATCHDOG dog=watchdog[dev];
		dog->failureCounter[rcv]++;
		if(dog->failureCounter[rcv] >= FAILURE_THRESHOLD)
		{
			blacklist_found(dev,rcv);
			dog->failureCounter[rcv] = 0;
		}
	}
	else
	{
		PWATCHDOG dog=watchdog[dev];
		dog->failureCounter[rcv]=0;
	}
	fn_NetSim_Packet_FreePacket(packet);
}
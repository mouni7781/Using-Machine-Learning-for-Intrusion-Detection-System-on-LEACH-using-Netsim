/************************************************************************************
 * Copyright (C) 2013                                                               *
 * TETCOS, Bangalore. India                                                         *
 *                                                                                  *
 * Tetcos owns the intellectual property rights in the Product and its content.     *
 * The copying, redistribution, reselling or publication of any or all of the       *
 * Product or its content without express prior written consent of Tetcos is        *
 * prohibited. Ownership and / or any other right relating to the software and all *
 * intellectual property rights therein shall remain at all times with Tetcos.      *
 *                                                                                  *
 * Aurhor:    Sanket Tapadia                                                      *
 *                                                                                  *
 * ---------------------------------------------------------------------------------*/

/**********************************IMPORTANT NOTES***********************************
1. This file contains the LEACH code. 
2. For this implementation of LEACH, the number of Clusters is fixed as 4 and all the
   4 clusters are equal.
   If the user wants to change it, then he/she must also change the static routing 
   for the Cluster Heads and the ClusterElement array accordingly.
3. To make 4 equal clusters. the number of sensors must be 4,16,36,64,100.
   Depending on the number of sensors, the ClusterElements array must be defined.
   Here, it has been defined and commented for 4,16,36,64,100 sensors. 
   Uncomment the one you want to use.
************************************************************************************/


#include "main.h"
#include "DSR.h"
#include "List.h"
#include "../BatteryModel/BatteryModel.h"
#include "../ZigBee/802_15_4.h"
#include "Pathrater.h"
#define NUMBEROFCLUSTERS 4
#define SIZEOFCLUSTERS 16					//SIZEOFCLUSTERS can be 1,4,9,16,25


static int CHcount[NUMBEROFCLUSTERS];
static int prevCH[NUMBEROFCLUSTERS];


//For 100 sensors and SIZEOFCLUSTERS = 25, uncomment this
//int ClusterElements[NUMBEROFCLUSTERS][SIZEOFCLUSTERS] = {{1,2,3,4,5,11,12,13,14,15,21,22,23,24,25,31,32,33,34,35,41,42,43,44,45},\
														{6,7,8,9,10,16,17,18,19,20,26,27,28,29,30,36,37,38,39,40,46,47,48,49,50},\
														{51,52,53,54,55,61,62,63,64,65,71,72,73,74,75,81,82,83,84,85,91,92,93,94,95},\
														{56,57,58,59,60,66,67,68,69,70,76,77,78,79,80,86,87,88,89,90,96,97,98,99,100}};

//For 64 sensors and SIZEOFCLUSTERS = 16, uncomment this
int ClusterElements[NUMBEROFCLUSTERS][SIZEOFCLUSTERS] = {{1,2,3,4,9,10,11,12,17,18,19,20,25,26,27,28},\
														   {5,6,7,8,13,14,15,16,21,22,23,24,29,30,31,32},\
														   {33,34,35,36,41,42,43,44,49,50,51,52,57,58,59,60},\
														   {37,38,39,40,45,46,47,48,53,54,55,56,61,62,63,64}};

//For 36 sensors and SIZEOFCLUSTERS = 9, uncomment this
//int ClusterElements[NUMBEROFCLUSTERS][SIZEOFCLUSTERS]= {{1,2,3,7,8,9,13,14,15},{4,5,6,10,11,12,16,17,18},{19,20,21,25,26,27,31,32,33},{22,23,24,28,29,30,34,35,36}};

//For 16 sensors and SIZEOFCLUSTERS = 4, uncomment this
//int ClusterElements[NUMBEROFCLUSTERS][SIZEOFCLUSTERS] = {{1,2,5,6},{3,4,7,8},{9,10,13,14},{11,12,15,16}};

//For 4 sensors and SIZEOFCLUSTERS = 1, uncomment this
//int ClusterElements[NUMBEROFCLUSTERS][SIZEOFCLUSTERS] = {{1},{2},{3},{4}};



int fn_NetSim_LEACH_CheckDestination(NETSIM_ID nDeviceId, NETSIM_ID nDestinationId)
//Function to check whether the Device ID is same as the Destination ID
{
	if(nDeviceId == nDestinationId)
		return 1;
	else
		return 0;
}

int fn_NetSim_LEACH_GetNextHop(NetSim_EVENTDETAILS* pstruEventDetails)
{
	int nextHop;
	NETSIM_ID nInterface;

	//int CH[NUMBEROFCLUSTERS] = {23,28,73,78};
	int CH[NUMBEROFCLUSTERS] = {19,22,43,46};
	//int CH[NUMBEROFCLUSTERS] = {8,11,26,29};
	//int CH[NUMBEROFCLUSTERS] = {6,7,10,11};
	//int CH[NUMBEROFCLUSTERS] = {1,2,3,4};

	int i;
	int ClusterId;
	
	//This for loop dynamically assigns the Cluster Heads based on their energy.
	//Comment this for loop to enable fixed cluster heads.
	for (i=0; i<NUMBEROFCLUSTERS ; i++)
	{
		if(CHcount[i] == 4)
		{
			CH [i]= fn_NetSim_LEACH_AssignClusterHead(i);
			prevCH[i] = CH[i];
			CHcount[i] = 0;
		}
		else
		{
			CHcount[i]++;
			if(prevCH[i] != 0)
				CH[i] = prevCH[i];
		}
	}

	
	//Static Routes defined for 4 Clusters.
	//If the sensor is the Cluster Head, it forwards it to the next Cluster Head or 
	//to the Sink.Otherwise, it forwards the packet to the Cluster Head of its cluster.
	if(pstruEventDetails->pPacket->nSourceId == pstruEventDetails->nDeviceId)
	//For the first hop
	{
		if(pstruEventDetails->nDeviceId == CH[0])
			nextHop = CH[2];
		else if(pstruEventDetails->nDeviceId == CH[1])
			nextHop = CH[3];
		else if(pstruEventDetails->nDeviceId == CH[2] || pstruEventDetails->nDeviceId == CH[3])
			nextHop = get_first_dest_from_packet(pstruEventDetails->pPacket);
		else
		{
			ClusterId = fn_NetSim_LEACH_IdentifyCluster(pstruEventDetails->nDeviceId);
			nextHop = CH[ClusterId];
		}
	}
	else
    {
		ClusterId = fn_NetSim_LEACH_IdentifyCluster(pstruEventDetails->nDeviceId);
		//if(ClusterId < 2)
		if (ClusterId == 0)
			nextHop = CH[1];
		else if (ClusterId == 2)
			nextHop = CH[3];
		else
			nextHop = get_first_dest_from_packet(pstruEventDetails->pPacket);
	}
		
	
	//Updating the Transmitter ID, Receiver ID and NextHopIP in the pstruEventDetails
	free(pstruEventDetails->pPacket->pstruNetworkData->szNextHopIp);
	pstruEventDetails->pPacket->pstruNetworkData->szNextHopIp = dsr_get_dev_ip(nextHop);
	pstruEventDetails->pPacket->nTransmitterId = pstruEventDetails->nDeviceId;
	pstruEventDetails->pPacket->nReceiverId = nextHop;
	

	return 1;
}

int fn_NetSim_LEACH_AssignClusterHead (int ClusterNumber)
//Function to dynamically assign Cluster Heads based on the sensors' remaining energy.
{
	int  ClusterHeadID,i,DeviceId;
	double ClusterHeadPower,MaximumPower;

	if(SIZEOFCLUSTERS % 2 ==0)
		DeviceId = ClusterElements[ClusterNumber][SIZEOFCLUSTERS/2];
	else
		DeviceId = ClusterElements[ClusterNumber][(SIZEOFCLUSTERS-1)/2];
	
	MaximumPower = battery_get_remaining_energy(WSN_PHY(DeviceId)->battery);
	ClusterHeadID = DeviceId;
		
	for(i=0; i<SIZEOFCLUSTERS; i++)
	{
		DeviceId = ClusterElements[ClusterNumber][i];
		if (fn_NetSim_DSR_MaliciousNode(DeviceId))
		{
			ClusterHeadPower = 9999999999;
			if (find_ip_in_blacklist(DeviceId, DEVICE_NWADDRESS(DeviceId, 1)))
			{
				ClusterHeadPower = 0;				
			}
		}
		else
			ClusterHeadPower = battery_get_remaining_energy(WSN_PHY(DeviceId)->battery);
		if(MaximumPower < ClusterHeadPower)
		{
		MaximumPower = ClusterHeadPower;
		ClusterHeadID = DeviceId;
		}
	}
	return ClusterHeadID;
}

int fn_NetSim_LEACH_IdentifyCluster(int DeviceId)
//Function to identify the cluster of the sensor.
{
	int i,j;
	for(i=0; i<NUMBEROFCLUSTERS; i++)
	{
		for(j=0; j<SIZEOFCLUSTERS; j++)
		{
			if(DeviceId == ClusterElements[i][j])
				return i;
		}
	}
}


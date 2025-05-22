/****************************************************
         This file contains code for generation of Malicious Node(SinkHole) for networks running DSR in Layer3.
		 This works only for UDP and not for TCP.
		 
		 
		 The function fn_NetSim_DSR_MaliciousNode(NetSim_EVENTDETAILS*) 
		 return 1 when the deviceID is the malicious node which is mentioned in the if statement in the function definition.

		
		 The function fn_NetSim_DSR_MaliciousProcessSourceRouteOption(NetSim_EVENTDETAILS*)
		 does not call the NetworkOut Event and destroys the packet.

		 Code Flow - 
		 If The Node is a Malicious Node, Then when a Route Request is Received, the Function adds the route from itself 
		 to the target in the route cache and sends a false route reply.
		 When a malicious node receives a data packet, it gives acknowledge reply and frees the packet.
		
		 
*****************************************************/


	/* Malicious Node */


#include "main.h"
#include "DSR.h"
#include "List.h"

//#define MALICIOUS_NODE1 2
#define array_size(_arr) (sizeof(_arr)/sizeof(_arr[0]))

bool fn_NetSim_DSR_MaliciousNode(NETSIM_ID,double);
int fn_NetSim_DSR_MaliciousRouteAddToCache(NetSim_EVENTDETAILS*);
int fn_NetSim_DSR_MaliciousProcessSourceRouteOption(NetSim_EVENTDETAILS*);



typedef struct stru_malicious_node
{
	NETSIM_ID id; //Malicious node id
	double time; //In microsec. Time at which node become malicious
}MALICIOUS_NODE,*PMALICIOUS_NODE;

/* format	{	{2,0},
*				{5,1000000},
*				{8,900}
*			};
*/
MALICIOUS_NODE malicious_node[] = { {11,0.0} }; //Example:Node 11 is malicious node after time 0 sec.
	

bool fn_NetSim_DSR_MaliciousNode(NETSIM_ID dev,double time)
{
	int i;
	#ifdef _IDS_
	#ifdef _IDS_METRIC_INIT
	int n;
	//mal_dev_count=(sizeof(malicious_node)/(sizeof(unsigned short int)+sizeof(double)));
	mal_dev_count=sizeof(malicious_node)/16;
	
	for(n=0;n<mal_dev_count;n++)
	{
	mal_devid[n]=malicious_node[n].id;
	mal_start_time[n]=malicious_node[n].time;
	
	}
	#undef _IDS_METRIC_INIT
	#endif
	#endif
	
	for(i=0;i<array_size(malicious_node);i++)
	{
		if(dev == malicious_node[i].id && time >= malicious_node[i].time)
		{
			return true;
		}
	}
	return false;
}






int fn_NetSim_DSR_MaliciousProcessSourceRouteOption(NetSim_EVENTDETAILS* pstruEventDetails)
{
	NetSim_PACKET* packet = pstruEventDetails->pPacket;
	fn_NetSim_Packet_FreePacket(pstruEventDetails->pPacket);
	return 0;
}

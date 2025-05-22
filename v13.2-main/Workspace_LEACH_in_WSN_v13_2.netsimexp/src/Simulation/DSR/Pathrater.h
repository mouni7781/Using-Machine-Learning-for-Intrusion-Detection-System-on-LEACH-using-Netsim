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
#ifdef _IDS_
#define _NETSIM_PATHRATER_ //Uncomment this to run pathrater
#endif


#include "List.h"
#ifndef _NETSIM_PATHRATER_H_
#define _NETSIM_PATHRATER_H_
#ifdef  __cplusplus
extern "C" {
#endif
	
	typedef struct stru_blacklist
	{
		NETSIM_IPAddress ip;
		_ele* ele;
	}BLACKLIST,*PBLACKLIST;
#define BLACKLIST_ALLOC() (PBLACKLIST)list_alloc(sizeof(BLACKLIST),offsetof(BLACKLIST,ele))
#define BLACKLIST_NEXT(_blacklist) _blacklist=(PBLACKLIST)LIST_NEXT(_blacklist)
#define BLACKLIST_FREE_ALL(_blacklist) while(_blacklist) LIST_FREE((void**)&_blacklist,_blacklist)
#define BLACKLIST_ADD(_list,_black) LIST_ADD_LAST((void**)&_list,_black)

	//Function prototype

	//Initiallize the pathrater. Must be called in init function.
	void pathrater_init(); 

	//Close the pathrater. Must be called in finish function.
	void pathrater_close();

	//Add the ip to blacklist
	void add_to_blacklist(NETSIM_ID dev_id,NETSIM_IPAddress ip);

	//Verify the route reply.
	bool verify_route_reply(NETSIM_ID dev_id,DSR_RREP_OPTION* rrep);

	bool is_malicious(NETSIM_ID dev,double time);
	int Malicious_AddRouteToCache(NetSim_EVENTDETAILS* pstruEventDetails);
	int Malicious_DropPacket(NetSim_EVENTDETAILS* pstruEventDetails);

#ifdef  __cplusplus
}
#endif
#endif

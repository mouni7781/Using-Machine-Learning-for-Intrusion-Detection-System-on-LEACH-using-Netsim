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
In this code, (WATCHDOG_TIME) a watchdog timer is set for duration
of 2 sec. User can modify it.
The nodes wait for the duration of this timer to detect if the next 
hop node is malicious or not by checking if it broadcasts the packet
further to the destination node.

****************************************************/
#include "../DSR/DSR.h"
#ifdef _IDS_
#define _NETSIM_WATCHDOG_ //Uncomment this to run watchdog
#endif
#include "List.h"
#ifndef _NETSIM_WATCHDOG_H_
#define _NETSIM_WATCHDOG_H_
#ifdef  __cplusplus
extern "C" {
#endif

#ifdef _NETSIM_WATCHDOG_
#pragma comment(lib,"libDSR.lib")
	_declspec(dllexport) void blacklist_found(NETSIM_ID dev_id,NETSIM_ID black_id);
#endif

//watchdog timer is set for 2 sec
#define WATCHDOG_TIME		2*SECOND 
//the number of times packet is resend before declaring malicious node
#define FAILURE_THRESHOLD	20		 

	typedef struct stru_watchdog
	{
		unsigned int* failureCounter;
		NetSim_PACKET* sent_packet;
	}WATCHDOG,*PWATCHDOG;


	//Function prototype
	void watchdog_init();
	void watchdog_close();
	void add_to_sent_packet(NETSIM_ID dev,NetSim_PACKET* packet);
	void add_watchdog_timer(NETSIM_ID dev,NetSim_PACKET* packet);
	void watchdog_timer_execute();


#ifdef  __cplusplus
}
#endif
#endif
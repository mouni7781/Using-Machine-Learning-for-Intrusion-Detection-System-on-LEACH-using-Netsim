#pragma once
/************************************************************************************
* Copyright (C) 2020																*
* TETCOS, Bangalore. India															*
*																					*
* Tetcos owns the intellectual property rights in the Product and its content.		*
* The copying, redistribution, reselling or publication of any or all of the		*
* Product or its content without express prior written consent of Tetcos is			*
* prohibited. Ownership and / or any other right relating to the software and all	*
* intellectual property rights therein shall remain at all times with Tetcos.		*
*																					*
* This source code is licensed per the NetSim license agreement.					*
*																					*
* No portion of this source code may be used as the basis for a derivative work,	*
* or used, for any purpose other than its intended use per the NetSim license		*
* agreement.																		*
*																					*
* This source code and the algorithms contained within it are confidential trade	*
* secrets of TETCOS and may not be used as the basis for any other software,		*
* hardware, product or service.														*
*																					*
* Author:    Shashi Kant Suman	                                                    *
*										                                            *
* ----------------------------------------------------------------------------------*/
#ifndef _NETSIM_EMULATIONLIB_H_
#define _NETSIM_EMULATIONLIB_H_
#ifdef  __cplusplus
extern "C" {
#endif

#ifndef __IN__ 
#define __IN__
#endif

#ifndef __OUT__
#define __OUT__
#endif
#pragma comment(lib,"EmulationLib.lib")

#pragma region _PACP_HANDLES
#define HANDLE_COUNT		4

#define ALL_HANDLES			0
#define DISPATCH_HANDLE		1
#define REINJECT_HANDLE		2
#define NOTDISPATCHHANDLE	3

#define captureHandle		handles[ALL_HANDLES]
#define dispatchHandle		handles[DISPATCH_HANDLE]
#define reinjectHandle		handles[REINJECT_HANDLE]
#define notdispatchHandle	handles[NOTDISPATCHHANDLE]

	static char* pcapFileName[HANDLE_COUNT] =
	{ "ALL_NETWORK_PACKETS.pcap","DISPATCHED_TO_EMULATOR.pcap",
		"REINJECTED_FROM_EMULATOR.pcap","NOT_DISPATCHED_TO_EMULATOR.pcap"
	};

	static char* pipeName[HANDLE_COUNT] =
	{
		"NETSIM_ALL_CAPTURE","NETSIM_DISPATCHED","NETSIM_REINJECTED","NETSIM_NOT_DISPATCHED"
	};

	void* handles[HANDLE_COUNT];
	void create_emulation_pcap_handle(__IN__ int count, __IN__ int* pcap_logfile_status,
									  __IN__ char** pcapFileName, __IN__ char** pipeName,
									  __OUT__ void** handles);
	void close_emulation_pcap_handle(__IN__ int count, __IN__ void** handles);
#pragma endregion

#ifdef  __cplusplus
}
#endif
#endif //_NETSIM_EMULATIONLIB_H_

#pragma once
/************************************************************************************
* Copyright (C) 2021																*
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
#ifndef _NETSIM_MOBILITYINTERFACE_H_
#define _NETSIM_MOBILITYINTERFACE_H_
#ifdef  __cplusplus
extern "C" {
#endif

#ifndef _NETSIM_MOBILITY_CODE_
#pragma comment(lib,"Mobility.lib")
#endif

	_declspec(dllexport) bool fnMobility_findIntersect(NetSim_COORDINATES* p1, NetSim_COORDINATES* q1,
													   NetSim_COORDINATES* p2, NetSim_COORDINATES* q2,
													   NetSim_COORDINATES* intersect);
	_declspec(dllexport) bool fnMobility_isPosInsideBuilding(NetSim_COORDINATES* pos, NETSIM_ID id);
	_declspec(dllexport) UINT fnMobility_isPosInsideAnyBuilding(NetSim_COORDINATES* pos);
	_declspec(dllexport) bool fnMobility_findIntersectionPointbyBuilding(NetSim_COORDINATES* p1, NetSim_COORDINATES* p2,
																		 UINT id, NetSim_COORDINATES* ret);
	_declspec(dllexport) double fnMobility_findIndoorDistance(NetSim_COORDINATES* p1, NetSim_COORDINATES* p2);

#ifdef  __cplusplus
}
#endif
#endif /* _NETSIM_MOBILITYINTERFACE_H_ */
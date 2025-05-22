#pragma once
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

#ifndef _NETSIM_BATTERY_MODEL_H_
#define _NETSIM_BATTERY_MODEL_H_
#ifdef  __cplusplus
extern "C" {
#endif

#ifndef _BATTERY_MODEL_CODE_
#pragma comment(lib,"BatteryModel.lib")
	typedef void* ptrBATTERY;
#endif

	_declspec(dllexport) ptrBATTERY battery_find(NETSIM_ID d,
												 NETSIM_ID in);
	_declspec(dllexport) void battery_add_new_mode(ptrBATTERY battery,
												   int mode,
												   double current,
												   char* heading);
	_declspec(dllexport) ptrBATTERY battery_init_new(NETSIM_ID deviceId,
													 NETSIM_ID interfaceId,
													 double initialEnergy,
													 double voltage,
													 double dRechargingCurrent);
	_declspec(dllexport) bool battery_set_mode(ptrBATTERY battery,
											   int mode,
											   double time);
	_declspec(dllexport) void battery_animation();
	_declspec(dllexport) void battery_metrics(PMETRICSWRITER metricsWriter);
	_declspec(dllexport) double battery_get_remaining_energy(ptrBATTERY battery);
	_declspec(dllexport) double battery_get_consumed_energy(ptrBATTERY battery, int mode);

#ifdef  __cplusplus
}
#endif
#endif //_NETSIM_BATTERY_MODEL_H_
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

/*		   ,~~.																		*
 *		  (  6 )-_,																	*
 *	 (\___ )=='-'
 *	  \ .   ) )
 *	   \ `-' /
 *	~'`~'`~'`~'`~
 *
 *                                                                                  *
 * ---------------------------------------------------------------------------------*/
#ifndef _NETSIM_QUEUE_H_
#define _NETSIM_QUEUE_H_
#ifdef  __cplusplus
extern "C" {
#endif
#pragma comment(lib,"NetSim_utility.lib")
	typedef void* ptrQUEUE;

	_declspec(dllexport) ptrQUEUE queue_init(double size);
	_declspec(dllexport) void queue_free(ptrQUEUE q);
	_declspec(dllexport) bool queue_enqueue(ptrQUEUE queue, const void* mem, double size);
	_declspec(dllexport) void* queue_get_head_ptr(ptrQUEUE queue);
	_declspec(dllexport) void* queue_dequeue(ptrQUEUE queue);
	_declspec(dllexport) double queue_get_size(ptrQUEUE queue);
	_declspec(dllexport) bool queue_is_empty(ptrQUEUE queue);
#define queue_is_not_empty(q) (queue_is_empty((q)) == false)
	_declspec(dllexport) double queue_get_head_size(ptrQUEUE queue);
	_declspec(dllexport) void queue_update_head_size(ptrQUEUE queue, double newSize);

#ifdef  __cplusplus
}
#endif
#endif //_NETSIM_QUEUE_H_

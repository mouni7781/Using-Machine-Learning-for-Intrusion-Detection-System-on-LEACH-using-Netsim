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
#include "DSR.h"
#include "List.h"
/**
This function adds a route to the route cache. This is the entire list of address from
the device to the target in sequence
*/
int fn_NetSim_DSR_UpdateRouteCache(unsigned int length,
	NETSIM_IPAddress* address,
	double dTime)
{
	NETSIM_ID nDeviceId = pstruEventDetails->nDeviceId;
	DSR_DEVICE_VAR* devVar = DSR_DEV_VAR(nDeviceId);
	DSR_ROUTE_CACHE* cache = ROUTECACHE_ALLOC();
	unsigned int count = 0;
	while(count<length)
	{
		if(!IP_COMPARE(address[count],dsr_get_curr_ip()))
			break;
		count++;
	}
	count = length-count;
	cache->nLength = count;
	cache->dTimeOutTime = dTime+ROUTE_CACHE_TIMEOUT;
	cache->address = calloc(count,sizeof* cache->address);
	while(count--)
	{
		cache->address[count] = IP_COPY(address[--length]);
	}
	LIST_ADD_LAST(&devVar->pstruRouteCache,cache);
	return 0;
}
/**
This function searches the route cache for a route to the target from the device.
If route is found, it returns the route.
*/
DSR_ROUTE_CACHE* fn_NetSim_DSR_FindCache(DSR_DEVICE_VAR* devVar,NETSIM_IPAddress address,double dTime)
{
	DSR_ROUTE_CACHE* cache=devVar->pstruRouteCache;
	while(cache)
	{
		unsigned int nLoop;
		if(cache->dTimeOutTime <= dTime)
		{
			DSR_ROUTE_CACHE* temp = cache;
			cache = (DSR_ROUTE_CACHE*)LIST_NEXT(cache);
			LIST_FREE((void**)&devVar->pstruRouteCache,temp);
			continue; //cache expired
		}
		for(nLoop=0;nLoop<cache->nLength;nLoop++)
		{
			if(!IP_COMPARE(cache->address[nLoop],address))
			{
				cache->dTimeOutTime = dTime+ROUTE_CACHE_TIMEOUT;
				return cache;
			}
		}
		cache = LIST_NEXT(cache);
	}
	return NULL;
}
/**
If the route cache contains the address in the addList, it returns a false.
*/
bool fn_NetSim_DSR_ValidateRouteCache(DSR_ROUTE_CACHE* cache,NETSIM_IPAddress* addList,int count)
{
	unsigned int loop;
	int loop2;
	for(loop=0;loop<cache->nLength;loop++)
	{
		for(loop2=0;loop2<count;loop2++)
		{
			if(!IP_COMPARE(cache->address[loop],addList[loop2]))
				return false;
		}
	}
	return true;
}
/**
This function deletes an entry from the route cache.
*/
int fn_NetSim_DSR_DeleteEntryFromRouteCache(DSR_ROUTE_CACHE** ppcache,
	NETSIM_IPAddress ip1,
	NETSIM_IPAddress ip2)
{
	DSR_ROUTE_CACHE* cache = *ppcache;
	while(cache)
	{
		unsigned int loop;
		int flag = 0;
		for(loop=0;loop<cache->nLength-1;loop++)
		{
			if(!IP_COMPARE(ip1,cache->address[loop]) &&
				!IP_COMPARE(ip2,cache->address[loop+1]))

			{
				unsigned int loop2;
				//cache found
				for(loop2=0;loop2<cache->nLength;loop2++)
					IP_FREE(cache->address[loop2]);
				free(cache->address);
				LIST_FREE((void**)ppcache,cache);
				cache=*ppcache;
				flag=1;
				break;
			}
		}
		if(flag)
			continue;
		cache = (DSR_ROUTE_CACHE*)LIST_NEXT(cache);
	}
	return 1;
}






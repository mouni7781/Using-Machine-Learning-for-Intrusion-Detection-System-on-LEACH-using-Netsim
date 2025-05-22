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
#define _BATTERY_MODEL_CODE_
#include "main.h"
#include "Animation.h"

#pragma comment(lib,"NetworkStack.lib")
#pragma comment(lib,"Metrics.lib")

typedef struct stru_mode
{
	int mode;
	double current;
	double consumedEnergy;
	char* heading;
}BATTERYMODE, *ptrBATTERYMODE;

typedef struct stru_battery
{
	NETSIM_ID deviceId;
	NETSIM_ID interfaceId;
	double initialEnergy;
	double consumedEnergy;
	double remainingEnergy;
	double voltage;
	double rechargingCurrent;
	int modeCount;
	ptrBATTERYMODE mode;

	int currentMode;
	double modeChangedTime;

	void* animHandle;
	struct stru_battery* next;
}BATTERY, *ptrBATTERY;

#include "BatteryModel.h"

//Function prototype
static void battery_add_to_animation(ptrBATTERY pb);

static ptrBATTERY firstBattery = NULL;
static ptrBATTERY lastBattery = NULL;

static void battery_add(ptrBATTERY b)
{
	if (firstBattery)
	{
		lastBattery->next = b;
		lastBattery = b;
	}
	else
	{
		firstBattery = b;
		lastBattery = b;
	}
}

static ptrBATTERYMODE battery_find_mode(ptrBATTERY b,
										int mode)
{
	int i;
	for (i = 0; i < b->modeCount; i++)
	{
		if (b->mode[i].mode == mode)
			return &b->mode[i];
	}
	return NULL;
}

_declspec(dllexport) ptrBATTERY battery_find(NETSIM_ID d,
											 NETSIM_ID in)
{
	ptrBATTERY db = NULL;
	ptrBATTERY t = firstBattery;
	while (t)
	{
		if (t->deviceId == d &&
			t->interfaceId == in)
			return t;

		if (t->deviceId == d &&
			t->interfaceId == 0)
			db = t;
		t = t->next;
	}
	return db;
}

_declspec(dllexport) void battery_add_new_mode(ptrBATTERY battery,
											   int mode,
											   double current,
											   char* heading)
{
	if (battery->modeCount)
		battery->mode = realloc(battery->mode, (battery->modeCount + 1) * sizeof* battery->mode);
	else
		battery->mode = calloc(1, sizeof* battery->mode);
	battery->mode[battery->modeCount].current = current;
	battery->mode[battery->modeCount].mode = mode;
	battery->mode[battery->modeCount].heading = _strdup(heading);
	battery->mode[battery->modeCount].consumedEnergy = 0;
	battery->modeCount++;
}

_declspec(dllexport) ptrBATTERY battery_init_new(NETSIM_ID deviceId,
												 NETSIM_ID interfaceId,
												 double initialEnergy,
												 double voltage,
												 double dRechargingCurrent)
{
	ptrBATTERY nb = battery_find(deviceId, interfaceId);
	if (nb)
	{
		if (!interfaceId)
			return nb;

		if (nb->interfaceId)
			return nb;
	}
	nb = (ptrBATTERY)calloc(1, sizeof* nb);
	nb->deviceId = deviceId;
	nb->interfaceId = interfaceId;
	nb->initialEnergy = initialEnergy;
	nb->remainingEnergy = initialEnergy;
	nb->voltage = voltage;
	nb->rechargingCurrent = dRechargingCurrent;
	battery_add(nb);
	battery_add_new_mode(nb, 0, 0, NULL);
	battery_add_to_animation(nb);
	return nb;
}

_declspec(dllexport) void battery_free(NETSIM_ID deviceId,
									   NETSIM_ID interfaceId,
									   double time)
{
	ptrBATTERY nb = battery_find(deviceId, interfaceId);
	if (nb)
	{
		battery_set_mode(nb, 0, time);
		free(nb->mode);
		free(nb);
	}
}

_declspec(dllexport) bool battery_set_mode(ptrBATTERY battery,
										   int mode,
										   double time)
{
	bool ret;
	ptrBATTERYMODE pm = battery_find_mode(battery, battery->currentMode);
	double r = battery->voltage*battery->rechargingCurrent*(time - battery->modeChangedTime) / 1000000;
	double c = battery->voltage*pm->current*(time - battery->modeChangedTime) / 1000000;
	battery->remainingEnergy += r;
	battery->modeChangedTime = time;
	if (c > battery->remainingEnergy)
	{
		pm->consumedEnergy += battery->remainingEnergy;
		battery->consumedEnergy += battery->remainingEnergy;
		battery->remainingEnergy = 0;
		battery->currentMode = 0;
		ret = false;
	}
	else
	{
		pm->consumedEnergy += c;
		battery->consumedEnergy += c;
		battery->remainingEnergy -= c;
		battery->currentMode = mode;
		ret = true;
	}

	double per = battery->remainingEnergy * 100;
	per /= battery->initialEnergy;
	if (battery->animHandle)
		animation_add_new_entry(battery->animHandle,
								ANIM_BATTERY,
								"%d,%lf,%lf,",
								DEVICE_CONFIGID(battery->deviceId),
								per,
								time);
	return ret;
}

static ANIM_HANDLE handle;
static void battery_add_to_animation(ptrBATTERY pb)
{
	static bool isCalled = false;
	if (!handle)
	{
		handle = anim_add_new_menu(NULL,
								   "Battery Power",
								   false,
								   false,
								   false,
								   BUFSIZ,
								   ANIMFILETYPE_GENERIC);
	}
	char name[BUFSIZ];
	sprintf(name, "%s_BatteryPower", DEVICE_NAME(pb->deviceId));
	pb->animHandle =
		anim_add_new_menu(handle,
						  name,
						  true,
						  false,
						  true,
						  BUFSIZ,
						  ANIMFILETYPE_BATTERY);
}

_declspec(dllexport) void battery_animation()
{
	ptrBATTERY b = firstBattery;
	while (b)
	{
		battery_add_to_animation(b);
		b = b->next;
	}
}

typedef struct stru_batteryMetrics
{
	char* mode;
	int index;
	int count;
	double* cols;
	struct stru_batteryMetrics* next;
}BATTERYMETRICS, *ptrBATTERYMETRICS;
static ptrBATTERYMETRICS batteryMetrics = NULL;

static bool isIncludedInMetricsHeading(ptrBATTERYMODE m)
{
	ptrBATTERYMETRICS bm = batteryMetrics;
	while (bm)
	{
		if (!_stricmp(m->heading, bm->mode))
			return true;
		bm = bm->next;
	}
	return false;
}

static void addInMetricsHeading(ptrBATTERYMODE m)
{
	ptrBATTERYMETRICS bm = batteryMetrics;
	ptrBATTERYMETRICS pbm = NULL;
	int index = 0;
	while (bm)
	{
		index = bm->index;
		pbm = bm;
		bm = bm->next;
	}
	index++;
	ptrBATTERYMETRICS b = calloc(1, sizeof* b);
	b->index = index;
	b->mode = m->heading;
	if (pbm)
		pbm->next = b;
	else
		batteryMetrics = b;
}

static void prepare_mode_list(char* buf)
{
	ptrBATTERY b = firstBattery;
	while (b)
	{
		int i;
		for (i = 0; i < b->modeCount; i++)
		{
			if (!b->mode[i].heading)
				continue;
			if (!isIncludedInMetricsHeading(&b->mode[i]))
			{
				addInMetricsHeading(&b->mode[i]);
				strcat(buf, b->mode[i].heading);
				strcat(buf, "#0,");
			}
		}
		b = b->next;
	}
}

static double get_value_from_battery(ptrBATTERY b, char* mode)
{
	int i;
	for (i = 0; i < b->modeCount; i++)
	{
		if (!b->mode[i].heading)
			continue;
		if (!_stricmp(b->mode[i].heading, mode))
			return b->mode[i].consumedEnergy;
	}
	return -1;
}

static void add_to_mode_list(ptrBATTERY b)
{
	ptrBATTERYMETRICS bm = batteryMetrics;
	while (bm)
	{
		if (bm->count)
			bm->cols = realloc(bm->cols, (bm->count + 1) * sizeof* bm->cols);
		else
			bm->cols = calloc(1, sizeof* bm->cols);

		bm->cols[bm->count] = get_value_from_battery(b, bm->mode);
		bm->count++;
		bm = bm->next;
	}
}

static void add_to_table(PMETRICSNODE table,
						 int index)
{
	ptrBATTERYMETRICS bm = batteryMetrics;
	while (bm)
	{
		add_table_row_formatted(true,
								table,
								"%lf,",
								bm->cols[index]);
		bm = bm->next;
	}
}

_declspec(dllexport) void battery_metrics(PMETRICSWRITER metricsWriter)
{
	ptrBATTERY b;
	static bool isCalled = false;
	if (isCalled)
		return;
	isCalled = true;

	if (!firstBattery)
		return; // No battery is added

	PMETRICSNODE menu = init_metrics_node(MetricsNode_Menu, "Battery model", NULL);
	PMETRICSNODE table = init_metrics_node(MetricsNode_Table, "Battery model", NULL);
	add_node_to_menu(menu, table);

	char modeList[BUFSIZ]="";
	sprintf(modeList, "Device Name#1,Initial energy(mJ)#1,Consumed energy(mJ)#1,Remaining Energy(mJ)#1,");
	prepare_mode_list(modeList);
	add_table_heading_special(table, modeList);
	b = firstBattery;
	while (b)
	{
		add_to_mode_list(b);
		b = b->next;
	}

	int index = 0;
	b = firstBattery;
	while (b)
	{
		add_table_row_formatted(false, table,
								"%s,%lf,%lf,%lf,",
								DEVICE_NAME(b->deviceId),
								b->initialEnergy,
								b->consumedEnergy,
								b->remainingEnergy);
		add_to_table(table, index);
		b = b->next;
		index++;
	}
	write_metrics_node(metricsWriter, WriterPosition_Current, NULL, menu);
	delete_metrics_node(menu);
}

_declspec(dllexport) double battery_get_remaining_energy(ptrBATTERY battery)
{
	return battery->remainingEnergy;
}

_declspec(dllexport) double battery_get_consumed_energy(ptrBATTERY battery, int mode)
{
	if (!mode)
		return battery->consumedEnergy;
	int i;
	for (i = 0; i < battery->modeCount; i++)
	{
		if (battery->mode[i].mode == mode)
			return battery->mode[i].consumedEnergy;
	}
	return 0;
}


/************************************************************************************
 * Copyright (C) 2020                                                               *
 * TETCOS, Bangalore. India                                                         *
 *                                                                                  *
 * Tetcos owns the intellectual property rights in the Product and its content.     *
 * The copying, redistribution, reselling or publication of any or all of the       *
 * Product or its content without express prior written consent of Tetcos is        *
 * prohibited. Ownership and / or any other right relating to the software and all *
 * intellectual property rights therein shall remain at all times with Tetcos.      *
 *                                                                                  *
 * Author:    Shashi Kant Suman                                                       *
 *                                                                                  *
 * ---------------------------------------------------------------------------------*/
#ifndef _NETSIM_PLOT_H_
#define _NETSIM_PLOT_H_
#ifndef WIN32
#include <stdio.h>
#include "Linux.h"
#endif

typedef enum enum_plot_type
{
	Plot_Custom,
	Plot_Throughput,
	Plot_Buffer,
}PLOT_TYPE;

typedef struct stru_plot_data
{
	char* data;
	struct stru_plot_data* next;
}PLOT_DATA,*PPLOT_DATA;

typedef struct stru_plotData
{
	PPLOT_DATA read;
	PPLOT_DATA write;
}PLOTDATA,*ptrPLOTDATA;


typedef struct stru_NetSimPlot
{
	char* menuName;
	char* heading;
	char* xLabel;
	char* yLabel;

	UINT numY;
	char** plotName;

	PLOT_TYPE plotType;
	char* txtFileName;
	PLOTDATA plotData;
	bool isOver;

	HANDLE mutex;

}NETSIMPLOT,*PNETSIMPLOT;

/* Plot API */
_declspec(dllexport) PNETSIMPLOT fn_NetSim_Install_Metrics_Plot(PLOT_TYPE plotType,
																  char* menuName,
																  char* heading,
																  ...);
_declspec(dllexport) void add_plot_data_formatted(PNETSIMPLOT g, char* format, ...);
#endif

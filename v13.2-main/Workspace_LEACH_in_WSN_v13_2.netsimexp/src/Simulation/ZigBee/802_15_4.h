/************************************************************************************
* Copyright (C) 2020                                                                *
* TETCOS, Bangalore. India                                                          *
*                                                                                   *
* Tetcos owns the intellectual property rights in the Product and its content.      *
* The copying, redistribution, reselling or publication of any or all of the        *
* Product or its content without express prior written consent of Tetcos is         *
* prohibited. Ownership and / or any other right relating to the software and all   *
* intellectual property rights therein shall remain at all times with Tetcos.       *
*                                                                                   *
* Author:    Shashi Kant Suman						                                *
*                                                                                   *
* ---------------------------------------------------------------------------------*/

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  *
*                                                                                        *
*	THIS FILES CONTAINS IEEE 802.15.4 DATASTUCTURE WHICH HAS VARIABLES THAT ARE PROVIDED *
*   FOR USERS. BY MAKING USE OF THESE VARIABLES THE USER CAN CREATE THEIR OWN PROJECT.   *
*	                                                                                     *    
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  */
#ifndef _NETSIM_802_15_4_H_
#define _NETSIM_802_15_4_H_
#ifdef  __cplusplus
extern "C" {
#endif

#pragma comment(lib,"PropagationModel.lib")
#pragma comment(lib,"BatteryModel.lib")
//#pragma comment(lib,"Zigbee.lib")

	//Default Battery config parameter
#define BATTERY_RECHARGING_CURRENT_MA_DEFAULT	0
#define BATTERY_VOLTAGE_V_DEFAULT				0
#define BATTERY_INITIAL_ENERGY_DEFAULT			0
#define BATTERY_TRANSMITTING_CURRENT_MA_DEFAULT	0
#define BATTERY_RECEIVING_CURRENT_MA_DEFAULT	0
#define BATTERY_IDLEMODE_CURRENT_MA_DEFAULT		0
#define BATTERY_SLEEPMODE_CURRENT_MA_DEFAULT	0

#define dUnitSymbolTime  ((4*1000000.0)/250000) //Time taken to transfer 1 symbols. Each symbol is 4 bit long.
#define WSN_PHY(devId) ((IEEE802_15_4_PHY_VAR*)DEVICE_PHYVAR(devId,1))
#define WSN_MAC(devId) ((IEEE802_15_4_MAC_VAR*)DEVICE_MACVAR(devId,1))
#define WSN_LINK_ID(nDevice_Id,nInterfaceId,nConnectedDevId,nConnectedDevPortId) fn_NetSim_Stack_GetConnectedDevice(nDevice_Id,nInterfaceId,nConnectedDevId,nConnectedDevPortId);
#define WSN_MAC_HEADER(pPacket) ((IEEE802_15_4_HEADER*)PACKET_MACPROTOCOLDATA(pPacket))
#define CHANNEL_IDLE 0
#define CHANNEL_BUSY 1

#define MAX_TTL 255
#define SOURCEPORT 45
#define DESTINATIONPORT 50

	/* Typedef declaration of structures */

	typedef struct stru_propagation_info_for_list
	{
		NETSIM_ID TX_ID;
		NETSIM_ID RX_ID;
		NETSIM_ID TX_IF_ID;
		NETSIM_ID RX_IF_ID;
		PPROPAGATION_INFO Propagation_Info;
		_ptr_ele ele;
	}propagation_info_for_list, * ptrpropagation_info_for_list;

#define PPROPAGATION_INFO_ALLOC() (struct stru_propagation_info_for_list*)list_alloc(sizeof(struct stru_propagation_info_for_list),offsetof(struct stru_propagation_info_for_list,ele))
#define PPROPAGATION_INFO_ADD(info,e)		(LIST_ADD_LAST(&(info),(e)))
#define PPROPAGATION_INFO_NEXT(entity)	(LIST_NEXT(entity))
#define PPROPAGATION_INFO_REMOVE(ls, mem) (LIST_FREE((void**)(ls),(mem)))

	PPROPAGATION_INFO find_propagation_info(NETSIM_ID t, NETSIM_ID ti, NETSIM_ID r, NETSIM_ID ri);
	void fn_NetSim_ieee802_15_4_PropagationInit();
	int fn_NetSim_IEEE802_15_4_FreePropagationInfo();

	typedef struct stru_IEEE802_15_4_Header IEEE802_15_4_HEADER;
	typedef struct stru_IEEE802_15_4_MacVar IEEE802_15_4_MAC_VAR;
	typedef struct stru_IEEE802_15_4_PhyVar IEEE802_15_4_PHY_VAR, *ptrIEEE802_15_4_PHY_VAR;
	typedef struct stru_BeaconFrame BEACONFRAME;
	typedef struct stru_SuperFrame SUPERFRAME;
	typedef struct stru_Position_2D POS_2D;
	typedef struct struChannel CHANNELS;
	typedef struct stru_IEEE802_15_4_Metrics IEEE802_15_4_METRCIS;
	typedef struct stru_Metrics METRICS;

	/* Typedef declaration of enumerations */
	typedef enum enum_CCAMODE CCAMODE;
	typedef enum enum_IEEE_802_15_4_ControlPacket_Type IEEE802_15_4_PACKET_TYPE;
	typedef enum enum_BeaconMode BEACON_MODE;
	typedef enum enum_IEEE802_15_4_Subevent_Type SUB_EVENT;
	typedef enum enum_MacState MAC_STATE;
	/** Enumeration for Zigbee Control packet type */
	enum enum_IEEE_802_15_4_ControlPacket_Type
	{
		BEACON_FRAME = MAC_PROTOCOL_IEEE802_15_4*100+1,
		SLEEP_FRAME,
		ACK_FRAME,
	};
	/** Enumeration for Zigbee Super frame statue */
	enum enum_SuperFrame_Status
	{
		BEACONTRANSMISSIONMODE,
		CAPMODE,
		CFPMODE,
		INACTIVEMODE,
	};

	/** Defining sub event type of IEEE 802.15.4*/
	enum enum_IEEE802_15_4_Subevent_Type
	{
		SUPERFRAME_EVENT = MAC_PROTOCOL_IEEE802_15_4*100+1,
		CARRIERSENSE_START,
		CARRIERSENSE_END,
		ACK_TIMEOUT,			
		BEACON_TRANSMISSION, 
		GOFORSLEEP,			
		BEACON_TRANSMISSION_END, 
		CAP_END,				
		CFP_END,			
		ACK_EVENT,			
		CSMA_DATATRANSFER,	
		CHANGE_RX,			
		CHANGE_RXANDSENDDATA,
		UPDATE_MEDIUM,
		WATCHDOG_TIMER,	
	};
	/** Enumeration for Zigbee CCA Mode */
	enum enum_CCAMODE
	{
		CARRIERSENSE,
		ENERGYABOVETHERESHOLD,
		CARRIERSENSE_AND_ENERGYABOVETHERESHOLD,
		CARRIERSENSE_OR_ENERGYABOVETHERESHOLD,
	};
	/** Enumeration for Beacon mode (Enable or disable) */
	enum enum_BeaconMode
	{
		BEACON_DISABLE,
		BEACON_ENABLE,
	};
	
	/** Enumeration for MAc state */
	enum enum_MacState
	{
		IDLE,
		BACKOFF_MODE,
		CCA_MODE,
		TX_MODE,
		RX_MODE,
		OFF,
	};

	enum enum_IEEE802_15_4_isConfigured
	{
		DISABLE=0,
		ENABLE=1,
	};

	NETSIM_ID nGlobalPANCoordinatorId;
	/** Structure for IEEE 802.15.4 */
	struct stru_IEEE802_15_4_Header
	{
		//int n_NC_ChannelId;
		int nAckRequestFlag;
		struct stru_BeaconFrame* pstruBeaconFrame;
	};
	
	/** Data structure for Beacon frame */
	struct stru_BeaconFrame
	{
		int nBeaconId;
		int nSuperFrameId;
		int nBeaconTime;
		double dPayload;
		double dOverhead;
		double dFrameSize;
	};

	/** Data structure for Super frame */
	struct stru_SuperFrame
	{
		int nSuperFrameId;
		int nTimeSlotCount;
		int nSuperFrameStatus;
		double dSuperFrameStartTime;
		double dSuperFrameLength;
		double dActivePeriodLength;
		double dInactivePeriodLength;
		double dTimeSlotLength;
		double dCAPLength;
		double dCFPLength;
		double dBeaconLength;
		BEACONFRAME* pstruBeacon;
		struct stru_SuperFrame* pstruNextSuperFrame;
	};
	SUPERFRAME* pstruSuperFrame;

	/// Stores the information of MAC layer.
	struct stru_IEEE802_15_4_MacVar
	{
		ptrSOCKETINTERFACE sock;
		MAC_STATE nNodeStatus;
		int nAckRequestFlag;
		BEACON_MODE nBeaconMode;
		/// Range = 0 - 15 ; Default = 15 
		int nMacBeaconOrder;	
		/// Range = 0 - 15 ;  Default = 15 
		int nMacSuperframeOrder;
		/// 15.36 ms for 2.4 GHz
		double dBaseSuperFrameDuration; 
		/// Enable or Disable
		int nBatteryLifeExtension;	
		/// Range = 3 - 8 ; Default = 5 
		int nMacMaxBE;				
		///  Range = 0 - 5 ; Default = 4 
		int nMacMaxCSMABackoff;		
		/// Range = 0 - nMacMaxBE ; Default = 3     
		int nMacMinBE;				
		/// Range = 0 - 7 ;   Default = 3
		int nMacMaxFrameRetries;	
		/// 20 Symbols fixed
		int aUnitBackoffPeriod;		
		int nMinCAPLength;

		/// specific to simulation
		int nBackoffExponent;
		NetSim_PACKET *pstruBuffer;
		int nRetryCount;
		/// To perform CCA in Slotted CSMACA 
		int nContentionWindow; 
		int nNoOfBackOff;
		int nBeaconReceivedFlag;
		int nLastBeaconId;
		int nChannelNumber;
	};

	/// Stores information of physical layer.
	struct stru_IEEE802_15_4_PhyVar
	{
		double dAntennaHeight;
		double dAntennaGain;
		unsigned long int ulSeed1, ulSeed2;
		/// -85 dbm default 
		double dReceiverSensivity; 
		/// -85 -10 = -95 dbm default 
		double dEDThreshold;		 
		double dTransmitterPower_mw;
		double dTransmitterRange_m;
		/// Working Frequency Range. In MHz
		double dFrequencyBand_MHz;	
		/// Data rate in kbps
		double dDataRate_kbps;		
		/// Chip Rate in mcps
		double dChipRate_mcps;
		double dSymobleRate_kSymbolsPS;
		char* pszModulationTechnique;
		double dMinLIFSPeriod_Symbols;
		double dMinSIFSPeriod_Symbols;
			
		/// 20 symbols fixed 
		int aUnitBackoffPeriod; 
		/// 12 symbols fixed 
		int aTurnaroundTime; 
		/// 3, 7, 10, 40 
		int phySHRDDuration; 
		/// 0.4, 1.6, 2, 8
		double phySymbolsPerOctet;  
		/// CCA Mode
		CCAMODE enumCCAMode;       
		PHY_TX_STATUS nRadioState;
		double dTotalReceivedPower;
		double macAckWaitDuration;
		int nSensorFlag;
		PHY_TX_STATUS nOldState;

		double d0;
		double pl_d0;

		void* battery;
		bool isDeviceOn;

		ptrpropagation_info_for_list propagation_info_list;
	};
	/** Data structure for channel */
	struct struChannel
	{
		/// Stores the channel number.
		int nChannelNumber;
		/// Stores the frequency.
		double dFrequency;
		/// Stores channel(IDLE,BUSY etc) status.
		int nChannelStatus;
		/// Device id to which it is allocated.
		int nAllocatedDeviceId;
		/// Type of the device to which it is allocated.
		int nAllocatedDeviceType;
		/// Count of the devices those have been assigned to the channel
		int nAssociatedDeviceCount;
		int nNoOfPacket;
	};
	CHANNELS* pstruChannelList;

	/// Structure for IEEE 802.15.4 Metrics 
	struct stru_IEEE802_15_4_Metrics
	{
		int nPacketTransmitted;
		int nPacketReceived;
		int nAckTransmitted;
		int nAckReceived;
		int nCCAAttempt;
		int nSuccessfulCCAAttempt;
		int nFailedCCA;
		double dTotalbackofftime;
		double  dAveragebackofftime;
		int nNumberofBackoffCall;
		int nBeaconTransmitted;
		int nBeaconReceived;
		int nBeaconForwarded;
		double dBeaconTime;
		double dCAPTime;
		double dCFPTime;
	};
	
	/// Data structure for IEEE 802.15.4 metrics which consists of sensor metrics and power model metrics also.
	struct stru_Metrics
	{
		IEEE802_15_4_METRCIS *pstruIEEE802_15_4_Metrics;
	};
	METRICS** pstruMetrics;

	//Propagation macro
#define GET_RX_POWER_dbm(tx, rx, time) (_propagation_get_received_power_dbm(find_propagation_info(tx, 1, rx, 1), time))
#define GET_RX_POWER_mw(tx,rx,time) (DBM_TO_MW(GET_RX_POWER_dbm(tx,rx,time)))


#define ZIGBEE_UNSLOTTED() fn_NetSim_Zigbee_UnslottedCSMACA()
#define ZIGBEE_SLOTTED() fn_NetSim_Zigbee_SlottedCSMACA()
#define ZIGBEE_CHANGERADIOSTATE(nDeviceId,nOldState,nNewState) fn_NetSim_Zigbee_ChangeRadioState(nDeviceId,nOldState,nNewState)
#define ZIGBEE_RECEIVEDPOWER(pstruPhyVar,dDistance,dReceivedPower,nLinkID) fn_NetSim_Zigbee_CalculateReceivedPower(pstruPhyVar,dDistance,dReceivedPower,nLinkID)
#define ZIGBEE_LOCATEBACKOFFBOUNDARY(dTime,dBoundaryTime,aUnitBackoffPeriod,pstruSuperFrame) fn_NetSim_Zigbee_LocateBackoffBoundary(dTime,dBoundaryTime,aUnitBackoffPeriod,pstruSuperFrame)
#define ZIGBEE_SUPERFRAME() fn_NetSim_Zigbee_SuperFrameInitialization(&pstruSuperFrame,nGlobalPANCoordinatorId,pstruMetrics)
#define ZIGBEE_BACKOFF(nBackoffExponent,dBackoffTime,nUnitBacoffPeriod) fn_NetSim_Zigbee_BackoffTimeCalculation(nBackoffExponent,dBackoffTime,nUnitBacoffPeriod,pstruMetrics)
#define ZIGBEE_CHANNELFORMATION() fn_NetSim_Zigbee_ChannelFormation(&pstruChannelList)
#define ZIGBEE_CCA(dInterferencePower,nCCA_Mode,dReceiverSensivity,dEDThreshold) fn_NetSim_Zigbee_CCA(dInterferencePower,nCCA_Mode,dReceiverSensivity,dEDThreshold,pstruMetrics)
#define ZIGBEE_SINR(SNR,dTotalReceivedPower,dReceivedPower) fn_NetSim_Zigbee_CalculateSINR(SNR,dTotalReceivedPower,dReceivedPower)

	/* Zigbee Function */
	int fn_NetSim_Configure_WSN_POWER(void*, int);


	/****************** NetWorkStack DLL functions declarations *****************************************/
	/** Function for configuring Zigbee parameters*/
	_declspec(dllexport) int fn_NetSim_Zigbee_Configure(void** var);
	int fn_NetSim_Zigbee_Configure_F(void** var);
	/** Function for Initializing Zigbee protocol */
	_declspec (dllexport) int fn_NetSim_Zigbee_Init(struct stru_NetSim_Network* NETWORK_Formal, NetSim_EVENTDETAILS* pstruEventDetails_Formal, char* pszAppPath_Formal, char* pszWritePath_Formal, int nVersion_Type, void** fnPointer);
	int fn_NetSim_Zigbee_Init_F(struct stru_NetSim_Network*, NetSim_EVENTDETAILS*, char*, char*, int, void** fnPointer);
	/** Function to run Zigbee protocol */
	_declspec (dllexport) int fn_NetSim_Zigbee_Run();

	/// Function to free the Zigbee protocol variable and Unload the primitives
	_declspec(dllexport) int fn_NetSim_Zigbee_Finish();
	int fn_NetSim_Zigbee_Finish_F();
	/// Return the sub event name with respect to the sub event number for writing event trace
	_declspec (dllexport) char *fn_NetSim_Zigbee_Trace(int nSubEvent);
	char *fn_NetSim_Zigbee_Trace_F(int nSubEvent);
	/// Function to free the allocated memory for the Zigbee packet
	_declspec(dllexport) int fn_NetSim_Zigbee_FreePacket(NetSim_PACKET* );
	int fn_NetSim_zigbee_FreePacket_F(NetSim_PACKET* );
	/// Function to copy the Zigbee packet from source to destination
	_declspec(dllexport) int fn_NetSim_Zigbee_CopyPacket(NetSim_PACKET* ,NetSim_PACKET* );
	int fn_NetSim_Zigbee_CopyPacket_F(NetSim_PACKET* ,NetSim_PACKET* );
	/// Function to write Zigbee Metrics into Metrics.txt
	_declspec(dllexport) int fn_NetSim_Zigbee_Metrics(PMETRICSWRITER metricsWriter);
	int fn_NetSim_Zigbee_Metrics_F(PMETRICSWRITER metricsWriter);

	//Backoff seed

	unsigned long ulBackoffSeed1;
	unsigned long ulBackoffSeed2;

	int fn_NetSim_Zigbee_UnslottedCSMACA();
	int fn_NetSim_Zigbee_SlottedCSMACA();
	bool fn_NetSim_Zigbee_ChangeRadioState(NETSIM_ID nDeviceId, PHY_TX_STATUS nOldState, PHY_TX_STATUS nNewState);
	double fn_NetSim_Zigbee_CalculateBER(double sinr);
	int fn_NetSim_Zigbee_LocateBackoffBoundary(double dTime, double* dBoundaryTime, int aUnitBackoffPeriod, SUPERFRAME* pstruSuperFrame);
	int fn_NetSim_Zigbee_SuperFrameInitialization(SUPERFRAME** ppstruSuperFrame,NETSIM_ID nGlobalPANCoordinatorId,METRICS** pstruMetrics);
	int fn_NetSim_Zigbee_BackoffTimeCalculation(int nBackoffExponent,double* dBackoffTime,int nUnitBacoffPeriod,METRICS** pstruMetrics);
	int fn_NetSim_Zigbee_CCA(double dInterferencePower, CCAMODE nCCA_Mode, double dReceiverSensivity, double dEDThreshold,METRICS** pstruMetrics);
	int fn_NetSim_Zigbee_CalculateSINR(double *SNR, double dTotalReceivedPower,double dReceivedPower);
	int fn_NetSim_Zigbee_ChannelFormation(CHANNELS** pstruChannelList);
	
#ifdef  __cplusplus
}
#endif
#endif

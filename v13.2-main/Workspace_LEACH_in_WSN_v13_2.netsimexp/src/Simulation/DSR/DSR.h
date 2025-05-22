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

/****************************************************
RFC 4728          The Dynamic Source Routing Protocol      February 2007
*****************************************************/
#define _SINKHOLE_
#define _IDS_
#ifndef _NETSIM_DSR_H_
#define _NETSIM_DSR_H_
#ifdef  __cplusplus
extern "C" {
#endif

#pragma comment(lib,"NetworkStack.lib")
#pragma comment(lib,"DSR.lib")
#ifdef _IDS_
#define _IDS_METRIC_INIT
	_declspec(dllexport)int mal_dev_count;
	_declspec(dllexport)int mal_devid[100];
	_declspec(dllexport)double mal_start_time[100];
	_declspec(dllexport)double mal_det_time[100];
#endif // _IDS_



	/* Packet Size */
#define DSR_OPTION_HEADER_SIZE 4
#define DSR_RREQ_SIZE_FIXED 8
#define DSR_RREQ_SIZE_IPV6_FIXED 20
#define DSR_RREP_SIZE_FIXED 7
#define DSR_SOURCEROUTE_SIZE_FIXED 4
#define DSR_RERR_SIZE_FIXED 12+4/* type specific */
#define DSR_ACK_OPTION_LEN 10
#define DSR_ACK_REQUEST_LEN 4

#define DSR_RREQ_OPT_LEN 6
#define DSR_DISCOVERY_HOP_LIMIT 255
#define DSR_MAX_REQUEST_REXMT 16
#define DSR_MaxMaintRexmt 2


#define NO_NEXT_HEADER 59 //rfc 1700
#define NonpropRequestTimeout	(30*	MILLISECOND)
#define DSR_REQUEST_PERIOD		(500*	MILLISECOND)
#define DSR_MAX_REQUEST_PERIOD	(10*	SECOND)
#define ROUTE_CACHE_TIMEOUT		(300*	SECOND)
#define BROADCAST_JITTER		(10*	MILLISECOND)
#define MAINT_HOLD_OFF_TIME		(250*	MILLISECOND)

#define isDSRPACKET(packet) (packet->nControlDataType/100 == NW_PROTOCOL_DSR)


	typedef enum enum_DSR_Subevent DSR_SUBEVENT;
	typedef enum enum_DSR_ControlPacket DSR_CONTROL_PACKET;
	typedef enum enum_DSR_OptionType DSR_OPTION_TYPE;

	typedef struct stru_DSR_SendBuffer DSR_SEND_BUFFER;
	typedef struct stru_DSR_RouteCache DSR_ROUTE_CACHE;
	typedef struct stru_DSR_MaintBuffer DSR_MAINT_BUFFER;
	typedef struct stru_DSR_DeviceVar DSR_DEVICE_VAR;

	typedef struct stru_DSR_OptionHeader DSR_OPTION_HEADER;
	typedef struct stru_NetSim_DSR_RREQ_Option DSR_RREQ_OPTION;
	typedef struct stru_NetSim_DSR_RREP_Option DSR_RREP_OPTION;
	typedef struct stru_NetSim_DSR_RERR_Option DSR_RERR_OPTION;
	typedef struct stru_NetSim_DSR_SourceRouteOption DSR_SOURCE_ROUTE_OPTION;
	typedef struct stru_DSR_Ack_Request_Option DSR_ACK_REQ_OPTION;
	typedef struct stru_DSR_Ack_Option DSR_ACK_OPTION;
	typedef struct stru_DSR_RouteRequestTable DSR_RREQ_TABLE;

	typedef struct stru_DSR_Primitives DSR_PRIMITIVES;

	///Enumeration for DSR Subevents 
	enum enum_DSR_Subevent
	{
		subevent_RREQ_TIMEOUT=NW_PROTOCOL_DSR*100+1,
		subevent_MAINT_TIMEOUT,
		subevent_PROCESS_RERR,
	};

	///Enumeration for DSR Control Packets.
	enum enum_DSR_ControlPacket
	{
		ctrlPacket_ROUTE_REQUEST=NW_PROTOCOL_DSR*100+1,
		ctrlPacket_ROUTE_REPLY,
		ctrlPacket_ROUTE_ERROR,
		ctrlPacket_ACK,
	};

	///Enumeration for DSR option types.
	enum enum_DSR_OptionType
	{
		optType_RouteRequest=1,
		optType_RouteReply=2,
		optType_RouteError=3,
		optType_Ack=32,
		optType_SourceRoute=96,
		optType_AckRequest=160,
	};


	/**
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	6.7.  DSR Source Route Option

	The DSR Source Route option in a DSR Options header is encoded as
	follows:

	0                   1                   2                   3
	0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|  Option Type  |  Opt Data Len |F|L|Reservd|Salvage| Segs Left |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|                           Address[1]                          |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|                           Address[2]                          |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|                              ...                              |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|                           Address[n]                          |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
	*/
	struct stru_NetSim_DSR_SourceRouteOption
	{		
		DSR_OPTION_TYPE nOptionType; /**<
									 nOptionType = 96
									 Nodes not understanding this option will drop the packet.*/

		unsigned int nOptDataLen:8; /**<		
									8-bit unsigned integer.  Length of the option, in octets,
									excluding the Option Type and Opt Data Len fields.  For the
									format of the DSR Source Route option defined here, this field
									MUST be set to the value (n * 4) + 2, where n is the number of
									addresses present in the Address[i] fields.
									*/

		unsigned int F:1; /**<		
						  Set to indicate that the first hop indicated by the DSR Source
						  Route option is actually an arbitrary path in a network
						  external to the DSR network; the exact route outside the DSR network 
						  is not represented in the DSR Source Route option.
						  Nodes caching this hop in their Route Cache MUST flag the
						  cached hop with the External flag.  Such hops MUST NOT be
						  returned in a Route Reply generated from this Route Cache
						  entry, and selection of routes from the Route Cache to route a
						  packet being sent SHOULD prefer routes that contain no hops
						  flagged as External.
						  */

		unsigned int L:1; /**<
						  Set to indicate that the last hop indicated by the DSR Source
						  Route option is actually an arbitrary path in a network
						  external to the DSR network; the exact route outside the DSR
						  network is not represented in the DSR Source Route option.
						  Nodes caching this hop in their Route Cache MUST flag the
						  cached hop with the External flag.  Such hops MUST NOT be
						  returned in a Route Reply generated from this Route Cache
						  entry, and selection of routes from the Route Cache to route a
						  packet being sent SHOULD prefer routes that contain no hops
						  flagged as External.
						  */

		unsigned int nReserved:4; /**<	
								  MUST be sent as 0 and ignored on reception.
								  */		
		unsigned int nSalvage:4; /**<	
								 A 4-bit unsigned integer.  Count of number of times that this
								 packet has been salvaged as a part of DSR routing.
								 */

		unsigned int nSegsLeft:4; /**<
								  Segments Left (Segs Left)
								  Number of route segments remaining, i.e., number of explicitly
								  listed intermediate nodes still to be visited before reaching
								  the final destination.
								  */

		NETSIM_IPAddress* Address; /**<		 
								   The sequence of addresses of the source route.  In routing and
								   forwarding the packet, the source route is processed as
								   described in Sections 8.1.3 and 8.1.5.  The number of addresses
								   present in the Address[1..n] field is indicated by the Opt Data
								   Len field in the option (n = (Opt Data Len - 2) / 4).
								   */
	};

	/**
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	6.2.  Route Request Option

	The Route Request option in a DSR Options header is encoded as
	follows:

	0                   1                   2                   3
	0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|  Option Type  |  Opt Data Len |         Identification        |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|                         Target Address                        |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|                           Address[1]                          |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|                           Address[2]                          |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|                              ...                              |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|                           Address[n]                          |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	*/
	struct stru_NetSim_DSR_RREQ_Option
	{		
		DSR_OPTION_TYPE nOptionType; /**<
									 nOptionType = 1
									 Nodes not understanding this option will ignore this
									 option.
									 */

		unsigned int nOptDataLen:8; /**<		
									8-bit unsigned integer.  Length of the option, in octets,
									excluding the Option Type and Opt Data Len fields.  MUST be set
									equal to (4 * n) + 6, where n is the number of addresses in the
									Route Request Option.
									*/

		unsigned int nIdentification:16; /**<		
										 A unique value generated by the initiator (original sender) of
										 the Route Request.  Nodes initiating a Route Request generate a
										 new Identification value for each Route Request, for example
										 based on a sequence number counter of all Route Requests
										 initiated by the node.<br/>

										 This value allows a receiving node to determine whether it has
										 recently seen a copy of this Route Request.  If this
										 Identification value (for this IP Source address and Target
										 Address) is found by this receiving node in its Route Request
										 Table (in the cache of Identification values in the entry there
										 for this initiating node), this receiving node MUST discard the
										 Route Request.  When a Route Request is propagated, this field
										 MUST be copied from the received copy of the Route Request
										 being propagated.
										 */

		NETSIM_IPAddress targetAddress;

		NETSIM_IPAddress* address; /**<		
								   Address[i] is the IPv4 address of the i-th node recorded in the
								   Route Request option.  The address given in the Source Address
								   field in the IP header is the address of the initiator of the
								   Route Discovery and MUST NOT be listed in the Address[i]
								   fields; the address given in Address[1] is thus the IPv4
								   address of the first node on the path after the initiator.  The
								   number of addresses present in this field is indicated by the
								   Opt Data Len field in the option (n = (Opt Data Len - 6) / 4).
								   Each node propagating the Route Request adds its own address to
								   this list, increasing the Opt Data Len value by 4 octets.
								   */
	};

	/**
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	6.3 Route Reply Option

	The Route Reply option in a DSR Options header is encoded as follows:
	0                   1                   2                   3
	0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|  Option Type  |  Opt Data Len |L|   Reserved  |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|                           Address[1]                          |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|                           Address[2]                          |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|                              ...                              |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|                           Address[n]                          |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	*/
	struct stru_NetSim_DSR_RREP_Option
	{		
		DSR_OPTION_TYPE nOptionType; /**<
									 Option Type = 2
									 Nodes not understanding this option will ignore this
									 option.
									 */

		unsigned int nOptDataLen:8; /**<	
									8-bit unsigned integer.  Length of the option, in octets,
									excluding the Option Type and Opt Data Len fields.  MUST be set
									equal to (4 * n) + 1, where n is the number of addresses in the
									Route Reply Option.
									*/

		unsigned int nLastHopExternal:1; /**<
										 Set to indicate that the last hop given by the Route Reply (the
										 link from Address[n-1] to Address[n]) is actually an arbitrary
										 path in a network external to the DSR network; the exact route
										 outside the DSR network is not represented in the Route Reply.
										 Nodes caching this hop in their Route Cache MUST flag the
										 cached hop with the External flag.  Such hops MUST NOT be
										 returned in a cached Route Reply generated from this Route
										 Cache entry, and selection of routes from the Route Cache to
										 route a packet being sent SHOULD prefer routes that contain no
										 hops flagged as External.
										 */

		unsigned int nReserved:7; /**<
								  MUST be sent as 0 and ignored on reception.
								  */

		NETSIM_IPAddress* Address; /**<
								   The source route being returned by the Route Reply.  The route
								   indicates a sequence of hops, originating at the source node
								   specified in the Destination Address field of the IP header of
								   the packet carrying the Route Reply, through each of the
								   Address[i] nodes in the order listed in the Route Reply, ending
								   at the node indicated by Address[n].  The number of addresses
								   present in the Address[1..n] field is indicated by the Opt Data
								   Len field in the option (n = (Opt Data Len - 1) / 4).
								   */
	};

	/**
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	6.4.  Route Error Option

	The Route Error option in a DSR Options header is encoded as follows:

	0                   1                   2                   3
	0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|  Option Type  |  Opt Data Len |   Error Type  |Reservd|Salvage|
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|                      Error Source Address                     |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|                   Error Destination Address                   |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	.                                                               .
	.                   Type-Specific Information                   .
	.                                                               .
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	*/
	struct stru_NetSim_DSR_RERR_Option
	{
		DSR_OPTION_TYPE nOptionType; /**<		
									 Nodes not understanding this option will ignore this
									 option.
									 */

		unsigned int nOptDataLen:8; /**<
									8-bit unsigned integer.  Length of the option, in octets,
									excluding the Option Type and Opt Data Len fields.
									<br/>
									For the current definition of the Route Error option,
									this field MUST be set to 10, plus the size of any
									Type-Specific Information present in the Route Error.  Further
									extensions to the Route Error option format may also be
									included after the Type-Specific Information portion of the
									Route Error option specified above.  The presence of such
									extensions will be indicated by the Opt Data Len field.
									When the Opt Data Len is greater than that required for
									the fixed portion of the Route Error plus the necessary
									Type-Specific Information as indicated by the Option Type
									value in the option, the remaining octets are interpreted as
									extensions.  Currently, no such further extensions have been
									defined.
									*/
		/**
		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		Error Type
		The type of error encountered.  Currently, the following type
		values are defined:
		1 = NODE_UNREACHABLE
		2 = FLOW_STATE_NOT_SUPPORTED
		3 = OPTION_NOT_SUPPORTED
		~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		*/
		enum
		{
			NODE_UNREACHABLE=1,
			FLOW_STATE_NOT_SUPPORTED,
			OPTION_NOT_SUPPORTED,
		}nErrorType;

		unsigned int nReserved:4; /**<
								  Reserved.  MUST be sent as 0 and ignored on reception.
								  */

		unsigned int nSalvage:4; /**<
								 A 4-bit unsigned integer.  Copied from the Salvage field in the
								 DSR Source Route option of the packet triggering the Route
								 Error.

								 The "total salvage count" of the Route Error option is derived
								 from the value in the Salvage field of this Route Error option
								 and all preceding Route Error options in the packet as follows:
								 the total salvage count is the sum of, for each such Route
								 Error option, one plus the value in the Salvage field of that
								 Route Error option.
								 */

		NETSIM_IPAddress errorSourceAddress; /**<
											 The address of the node originating the Route Error (e.g., the
											 node that attempted to forward a packet and discovered the link
											 failure).
											 */

		NETSIM_IPAddress errorDestinationAddress; /**<
												  The address of the node to which the Route Error must be
												  delivered.  For example, when the Error Type field is set to
												  NODE_UNREACHABLE, this field will be set to the address of the
												  node that generated the routing information claiming that the
												  hop from the Error Source Address to Unreachable Node Address
												  (specified in the Type-Specific Information) was a valid hop.
												  */

		void* TypeSpecificInformation; /**<
									   Information specific to the Error Type of this Route Error
									   message.
									   */

		//path
		unsigned int length;
		unsigned int nSegsLeft;
		NETSIM_IPAddress* Address;
	};

	/**
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	6.5.  Acknowledgement Request Option

	The Acknowledgement Request option in a DSR Options header is encoded
	as follows:

	0                   1                   2                   3
	0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|  Option Type  |  Opt Data Len |         Identification        |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	*/
	struct stru_DSR_Ack_Request_Option
	{

		DSR_OPTION_TYPE nOptionType; /**<		
									 Nodes not understanding this option will remove the
									 option and return a Route Error.
									 */

		unsigned int nOptionDataLen:8; /**<		
									   8-bit unsigned integer.  Length of the option, in octets,
									   excluding the Option Type and Opt Data Len fields.
									   */

		unsigned int nIdentification:16; /**<		
										 The Identification field is set to a unique value and is copied
										 into the Identification field of the Acknowledgement option
										 when returned by the node receiving the packet over this hop.
										 */
	};

	/**
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	6.6.  Acknowledgement Option

	The Acknowledgement option in a DSR Options header is encoded as
	follows:

	0                   1                   2                   3
	0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|  Option Type  |  Opt Data Len |         Identification        |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|                       ACK Source Address                      |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|                     ACK Destination Address                   |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	*/
	struct stru_DSR_Ack_Option
	{	

		DSR_OPTION_TYPE nOptionType; /**<		
									 Nodes not understanding this option will remove the
									 option.
									 */		
		unsigned int nOptionDataLen:8; /**<		
									   8-bit unsigned integer.  Length of the option, in octets,
									   excluding the Option Type and Opt Data Len fields.
									   */		
		unsigned int nIdentification:16; /**<	
										 Copied from the Identification field of the Acknowledgement
										 Request option of the packet being acknowledged.
										 */		
		NETSIM_IPAddress sourceAddress; /**<		
										The address of the node originating the acknowledgement.
										*/		
		NETSIM_IPAddress DestAddress; /**<		
									  The address of the node to which the acknowledgement is to be
									  delivered.
									  */
	};

	/**
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	6.1.  Fixed Portion of DSR Options Header

	The fixed portion of the DSR Options header is used to carry
	information that must be present in any DSR Options header.  This
	fixed portion of the DSR Options header has the following format:

	0                   1                   2                   3
	0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	|  Next Header  |F|   Reserved  |        Payload Length         |
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	.                                                               .
	.                            Options                            .
	.                                                               .
	+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
	*/
	struct stru_DSR_OptionHeader
	{		
		unsigned int nNextHeader:8; /**<		
									8-bit selector.  Identifies the type of header immediately
									following the DSR Options header.  Uses the same values as the
									IPv4 Protocol field [RFC1700].  If no header follows, then Next
									Header MUST have the value 59, "No Next Header" [RFC2460].
									*/

		unsigned int nFlowState:1;/**<		
								  Flag bit.  MUST be set to 0.  This bit is set in a DSR Flow
								  State header (Section 7.1) and clear in a DSR Options header.
								  */

		unsigned int nReserved:7; /**<		
								  MUST be sent as 0 and ignored on reception.
								  Payload Length
								  The length of the DSR Options header, excluding the 4-octet
								  fixed portion.  The value of the Payload Length field defines
								  the total length of all options carried in the DSR Options
								  header.
								  */

		unsigned int nPayloadLength:16;
		DSR_OPTION_TYPE optType; /**<		
								 Variable-length field; the length of the Options field is
								 specified by the Payload Length field in this DSR Options
								 header.  Contains one or more pieces of optional information
								 (DSR options), encoded in type-length-value (TLV) format (with
								 the exception of the Pad1 option described in Section 6.8).
								 */		
		void* options;
		DSR_ACK_REQ_OPTION* ackRequestOption;
	};
	/**	
	Every Device maintains a route cache which has an entry of the entire route to the 
	target.
	*/
	struct stru_DSR_RouteCache
	{		
		double dTimeOutTime; /**<
							 dTimeOutTime - allows an entry to be deleted if not used within some time.
							 address - list of addresses in a route cache
							 ele - next route cache
							 */	

		unsigned int F:1;
		unsigned int L:1;
		unsigned int nLength:8;
		NETSIM_IPAddress* address;
		struct element* ele;
	};
	/**	
	Packets are added to send buffer if device doesn't have route to target
	*/
	struct stru_DSR_SendBuffer
	{
		NETSIM_IPAddress target; ///< Target IP address
		double dTime;
		NetSim_PACKET* packet; ///< List of packets to a particular Dest IP
		struct element* ele; ///< Next SendBuffer
	};
	/**
	DSR Route Request Table
	An Entry is made in the Route Request Table when a Route request is sent
	*/
	struct stru_DSR_RouteRequestTable
	{
		unsigned int nTTL; ///< nTTL - Time to Live		
		double lastRequestTime; /**<
								The time that this node last originated a Route Request for that
								target node.
								*/		
		double dBackoff; /**<
						 The remaining amount of time before which this node MAY next
						 attempt at a Route Discovery for that target node
						 */
		unsigned int nCount;
		NETSIM_IPAddress target;
		unsigned int nIdentification:16; ///< Identification value
		//Netsim specific
		unsigned long long int nEventId;
		bool flag;
		struct element* ele; ///< ele - next RouteRequestTable
	};
	/**	
	When a packet is forwarded to next HOP, it is added to the Maintanence Buffer. This is 
	added when the device has the route to target.
	*/
	struct stru_DSR_MaintBuffer
	{		
		double dAckTime;
		int count;
		NETSIM_ID nextHop; ///< Next hop ID of a route
		NETSIM_ID source;
		NETSIM_ID dest;
		NetSim_PACKET* packetList; ///< packetList - list of packets sent to the destination
		struct element* ele; ///< Next MaintBuffer
	};
	/**
	DSR Metrics structure contains various metrics like rrepSent, rrepForwarded etc.
	*/
	struct stru_DSR_Metrics
	{
		unsigned int rreqSent;
		unsigned int rreqForwarded;
		unsigned int rrepSent;
		unsigned int rrepForwarded;
		unsigned int rerrSent;
		unsigned int rerrForwarded;
		unsigned int routeBreak;
		unsigned int packetTransmitted;
		unsigned int packetOrginated;
		unsigned int packetReceived;
		unsigned int packetDropped;
	};
	/**
	DSR Device Variable
	*/
	struct stru_DSR_DeviceVar
	{

		unsigned int nRREQIdentification;
		struct stru_DSR_RouteCache* pstruRouteCache; ///< List of routes to particular destination
		struct stru_DSR_SendBuffer* pstruSendBuffer; ///<  Buffer in which packets are added if route to target is not known.
		struct stru_DSR_RouteRequestTable* pstruRREQTable; ///< A table in which entry of RREQ initiated is made.
		enum
		{
			LINK_LAYER_ACK,
			NETWORK_LAYER_ACK,
		}AckType;
		DSR_MAINT_BUFFER* pstruMaintBuffer; ///< Buffer in which packets sent to next HOP are added.
		struct stru_DSR_Metrics dsrMetrics;
	};

	DSR_ROUTE_CACHE* fn_NetSim_DSR_FindCache(DSR_DEVICE_VAR* devVar,NETSIM_IPAddress address,double dTime);
	int fn_NetSim_DSR_GeneralPacketProcessing(NetSim_EVENTDETAILS* pstruEventDetails);
	bool fn_NetSim_DSR_CheckRouteFound(NETSIM_IPAddress destAddress,DSR_DEVICE_VAR* devVar,NETSIM_IPAddress* nextHop,double dTime,DSR_ROUTE_CACHE** cache);
	int fn_NetSim_DSR_AddSourceRouteOption(NetSim_PACKET* packet,DSR_ROUTE_CACHE* cache);
	int fn_NetSim_DSR_ProcessSourceRouteOption(NetSim_EVENTDETAILS* pstruEventDetails);
	int fn_NetSim_DSR_AddToMaintBuffer(NETSIM_ID nDeviceId,
		NetSim_PACKET* pstruPacket,
		double dTime);
	int fn_NetSim_DSR_MaintTimeout(NetSim_EVENTDETAILS* pstruEventDetails);
	int fn_NetSim_DSR_RetransmitBuffer(DSR_MAINT_BUFFER* maintBuffer,NetSim_EVENTDETAILS* pstruEventDetails);
	int fn_NetSim_DSR_EmptyMaintBuffer(NETSIM_ID nDeviceId,NETSIM_ID nextHop);
	bool fn_NetSim_DSR_AddToSendBuffer(DSR_SEND_BUFFER** sendBuffer,NetSim_PACKET* packet,double dTime);
	void fn_NetSim_DSR_EmptySendBuffer(NETSIM_IPAddress targetAddress,NETSIM_ID nDeviceId);
	void fn_NetSim_DSR_CheckSendBuffer(NETSIM_ID nDeviceId,double dTime);
	int fn_NetSim_DSR_TransmitPacketFromSendBuffer(DSR_SEND_BUFFER* sendBuffer,
		NETSIM_ID nDeviceId,
		double dTime);
	int fn_NetSim_DSR_UpdateRouteCache(unsigned int length,
		NETSIM_IPAddress* address,
		double dTime);
	DSR_ROUTE_CACHE* fn_NetSim_DSR_FindCache(DSR_DEVICE_VAR* devVar,NETSIM_IPAddress address,double dTime);
	bool fn_NetSim_DSR_ValidateRouteCache(DSR_ROUTE_CACHE* cache,NETSIM_IPAddress* addList,int count);
	int fn_NetSim_DSR_DeleteEntryFromRouteCache(DSR_ROUTE_CACHE** ppcache,
		NETSIM_IPAddress ip1,
		NETSIM_IPAddress ip2);
	int fn_NetSim_DSR_GenerateRREP(NetSim_PACKET* rreqPacket,NetSim_EVENTDETAILS* pstruEventDetails);
	int fn_NetSim_DSR_ProcessRREP(NetSim_EVENTDETAILS* pstruEventDetails);
	bool fn_NetSim_DSR_GenerateRREPUsingRouteCache(DSR_DEVICE_VAR* devVar,
		NetSim_PACKET* rreqPacket,
		double dTime,
		NetSim_EVENTDETAILS* pstruEventDetails);
	int fn_NetSim_DSR_ForwardRREP();
	int fn_NetSim_DSR_GenerateRERR(DSR_MAINT_BUFFER* maintBuffer,NetSim_EVENTDETAILS* pstruEventDetails);
	int fn_NetSim_DSR_ProcessRerr(NetSim_EVENTDETAILS* pstruEventDetails);
	NetSim_PACKET* fn_NetSim_DSR_InitRouteRequest(NETSIM_IPAddress target, DSR_RREQ_TABLE** rreqTable,
		NETSIM_ID nDeviceId,
		double dTime,
		NetSim_EVENTDETAILS* pstruEventDetails);
	int fn_NetSim_DSR_RREQTimeout(NetSim_EVENTDETAILS* pstruEventDetails);
	NetSim_PACKET* fn_NetSim_DSR_RetryRREQ(NETSIM_IPAddress targetAddress,
		NETSIM_ID nDeviceId,
		double dTime,
		NetSim_EVENTDETAILS* pstruEventDetails);
	int fn_NetSim_DSR_ProcessRREQ(NetSim_EVENTDETAILS* pstruEventDetails);
	int fn_NetSim_DSR_Add_Ack_request_Option(NetSim_PACKET* packet,NetSim_EVENTDETAILS* pstruEventDetails);
	int fn_NetSim_DSR_ProcessAckOption(NetSim_EVENTDETAILS* pstruEventDetails);
	int fn_NetSim_DSR_Process_AckRequestOption(NetSim_PACKET* packet,NetSim_EVENTDETAILS* pstruEventDetails);

#define DSR_PACKET_PROCESSING() fn_NetSim_DSR_GeneralPacketProcessing(pstruEventDetails)
#define DSR_CHECK_ROUTE_FOUND(destAddress,devVar,nextHop,dTime,ppcache) fn_NetSim_DSR_CheckRouteFound(destAddress,devVar,nextHop,dTime,ppcache)
	//Maint buffer
#define DSR_ADD_TO_MAINT_BUFFER(nDeviceId,pstruPacket,dTime) fn_NetSim_DSR_AddToMaintBuffer(nDeviceId,pstruPacket,dTime);
#define DSR_MAINT_TIMEOUT() fn_NetSim_DSR_MaintTimeout(pstruEventDetails)
#define DSR_RETRANSMIT_BUFFER(maintBuffer) fn_NetSim_DSR_RetransmitBuffer(maintBuffer,pstruEventDetails)
#define DSR_EMPTY_MAINT_BUFFER(nDeviceId,nextHop) fn_NetSim_DSR_EmptyMaintBuffer(nDeviceId,nextHop)
	//Route cache
#define DSR_UPDATE_ROUTE_CACHE(length,address,dTime) fn_NetSim_DSR_UpdateRouteCache(length,address,dTime)
#define DSR_FIND_CACHE(devVar,address,dTime) fn_NetSim_DSR_FindCache(devVar,address,dTime)
#define DSR_VALIDATE_CACHE(cache,addList,count) fn_NetSim_DSR_ValidateRouteCache(cache,addList,count)
#define DSR_DELETE_ENTRY_CACHE(ppcache,ip1,ip2) fn_NetSim_DSR_DeleteEntryFromRouteCache(ppcache,ip1,ip2)
	//Route error
#define DSR_GENERATE_RERR(maintBuffer) fn_NetSim_DSR_GenerateRERR(maintBuffer,pstruEventDetails)
#define DSR_PROCESS_RERR() fn_NetSim_DSR_ProcessRerr(pstruEventDetails)
	//Route reply
#define DSR_GENERATE_RREP(rreqPacket) fn_NetSim_DSR_GenerateRREP(rreqPacket,pstruEventDetails)
#define DSR_PROCESS_RREP() fn_NetSim_DSR_ProcessRREP(pstruEventDetails)
#define DSR_GENERATE_RREP_USING_CACHE(devVar,rreqPacket,dTime) fn_NetSim_DSR_GenerateRREPUsingRouteCache(devVar,rreqPacket,dTime,pstruEventDetails)
	//Route request
#define DSR_INIT_RREQ(target,pprreqTable,nDeviceId,dTime) fn_NetSim_DSR_InitRouteRequest(target,pprreqTable,nDeviceId,dTime,pstruEventDetails)
#define DSR_RREQ_TIMEOUT() fn_NetSim_DSR_RREQTimeout(pstruEventDetails)
#define DSR_RETRY_RREQ(targetAddress,nDeviceId,dTime) fn_NetSim_DSR_RetryRREQ(targetAddress,nDeviceId,dTime,pstruEventDetails)
#define DSR_PROCESS_RREQ() fn_NetSim_DSR_ProcessRREQ(pstruEventDetails)
	//Send buffer
#define DSR_ADD_TO_SEND_BUFFER(ppsendBuffer,packet,dTime) fn_NetSim_DSR_AddToSendBuffer(ppsendBuffer,packet,dTime)
#define DSR_EMPTY_SEND_BUFFER(targetAddress,nDeviceId) fn_NetSim_DSR_EmptySendBuffer(targetAddress,nDeviceId)
#define DSR_CHECK_SEND_BUFFER(nDeviceId,dTime) fn_NetSim_DSR_CheckSendBuffer(nDeviceId,dTime)
#define DSR_TRANSMIT_SEND_BUFFER(sendBuffer,nDeviceId,dTime) fn_NetSim_DSR_TransmitPacketFromSendBuffer(sendBuffer,nDeviceId,dTime)
	//Source route
#define DSR_ADD_SRC_ROUTE(packet,cache) fn_NetSim_DSR_AddSourceRouteOption(packet,cache)
#define DSR_PROCESS_SRC_ROUTE() fn_NetSim_DSR_ProcessSourceRouteOption(pstruEventDetails)
	//Network layer ack
#define DSR_ADD_ACK_REQUEST(packet) fn_NetSim_DSR_Add_Ack_request_Option(packet,pstruEventDetails)
#define DSR_PROCESS_ACK_REQUEST(packet) fn_NetSim_DSR_Process_AckRequestOption(packet,pstruEventDetails)
#define DSR_PROCESS_ACK()	fn_NetSim_DSR_ProcessAckOption(pstruEventDetails);

	//DSR Lib function declaration
	NetSim_PACKET* fn_NetSim_DSR_GenerateCtrlPacket(NETSIM_ID src,
		NETSIM_ID dest,
		NETSIM_ID recv,
		double dTime,
		DSR_CONTROL_PACKET type);
	double fn_NetSim_DSR_GetBroadCastJitter();
	int fn_NetSim_DSR_LinkLayerAck(NetSim_PACKET*);
	unsigned int fn_NetSim_DSR_FillAddress(DSR_SOURCE_ROUTE_OPTION* srcOption,DSR_ROUTE_CACHE* cache,NETSIM_IPAddress src,NETSIM_IPAddress dest);
	int fn_NetSim_DSR_Finish_F();
	DSR_RREQ_TABLE* getRREQTable(NETSIM_IPAddress target,DSR_RREQ_TABLE* table);

	void set_dsr_curr();
	NETSIM_IPAddress dsr_get_curr_ip();
	NETSIM_ID dsr_get_curr_if();
	NETSIM_IPAddress dsr_get_dev_ip(NETSIM_ID d);
	bool isDsrConfigured(NETSIM_ID d, NETSIM_ID in);
	
#define DSR_DEV_VAR(dev) ((DSR_DEVICE_VAR*)NETWORK->ppstruDeviceList[dev-1]->pstruNetworkLayer->RoutingVar)
#define DSR_RREQ_LEN(rreq) (int)(rreq->nOptDataLen-6)/4
#define DSR_RREP_LEN(rrep) (int)(rrep->nOptDataLen-3)/4+1


#define SENDBUFFER_ALLOC() list_alloc(sizeof(struct stru_DSR_SendBuffer),offsetof(struct stru_DSR_SendBuffer,ele))
#define RREQTABLE_ALLOC() (DSR_RREQ_TABLE*)list_alloc(sizeof(struct stru_DSR_RouteRequestTable),offsetof(struct stru_DSR_RouteRequestTable,ele))
#define ROUTECACHE_ALLOC() (DSR_ROUTE_CACHE*)list_alloc(sizeof(struct stru_DSR_RouteCache),offsetof(struct stru_DSR_RouteCache,ele))
#define MAINTBUFFER_ALLOC() (DSR_MAINT_BUFFER*)list_alloc(sizeof(DSR_MAINT_BUFFER),offsetof(DSR_MAINT_BUFFER,ele))

	_declspec(dllexport) int fn_NetSim_DSR_FreePacket(NetSim_PACKET* packet);
#ifdef  __cplusplus
}
#endif
#endif




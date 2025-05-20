import traffic
import csv
import joblib
import logging
import numpy as np
import pandas as pd
from traffic.data import airports
from geopy.distance import geodesic
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from traffic.core import Traffic, Flight
from traffic.core.mixins import PointMixin
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)

def opposite_runway(rwy: str):
    """
    Determine the opposite runway number given a runway, e.g. if rwy = 18L it returns 36R

    Parameters
    ----------
    rwy : str
        string number of the runway
    
    Returns
    ----------
    str
        string with the opposite number and side
    """
    if rwy is None:
        return print("No valid runway detected.")

    if rwy.isdigit():  # numeric rwy (no side letter)
        num = int(rwy)
        opposite_num = (num + 18) % 36 if num < 18 else (num - 18)
        return f"{opposite_num:02}"
    
    # rwy with a side letter
    num, side = int(rwy[:-1]), rwy[-1]
    opposite_num = (num + 18) % 36 if num < 18 else (num - 18)
    if opposite_num == 0: opposite_num = 36 # in case it's 0, turn into 36 because 00 does not exist!
    opposite_num = f"{opposite_num:02}"
    opposite_side = 'R' if side == 'L' else 'L' if side == 'R' else side

    return f"{opposite_num}{opposite_side}"

def retrieve_runway_information(airport: str, rwy: str):
    """
    Retrieves the geographic coordinates of both thresholds of a given runway, as well as the midpoint between them

    Given a runway identifier (e.g., "06" or "24") and an airport object, this function returns:
        - The coordinates of the specified runway threshold (`extreme1`)
        - The coordinates of the opposite threshold (`extreme2`)
        - The center point (`center`) between both thresholds

    Parameters
    ----------
    airport : str
        ICAO code of the airport
    rwy : str
        Runway identifier (e.g., "06", "24", "18R")

    Returns
    ----------
    tuple
        (extreme1, extreme2, center), each being a `PointMixin` object with latitude, longitude, and name attributes:
            - extreme1: Point at the specified runway threshold
            - extreme2: Point at the opposite threshold
            - center: Midpoint between both thresholds
    """    
    # DEFINE RUNWAY EXTREME POINTS #
    extreme1 = PointMixin()
    extreme2 = PointMixin()
    extreme1.name = rwy
    extreme2.name = opposite_runway(rwy)

    df = airport.runways.data
    extreme1_info = df[df.name == extreme1.name]
    extreme2_info = df[df.name == extreme2.name]

    extreme1.latitude = extreme1_info.latitude.iloc[0]
    extreme1.longitude = extreme1_info.longitude.iloc[0]
    extreme2.latitude = extreme2_info.latitude.iloc[0]
    extreme2.longitude = extreme2_info.longitude.iloc[0]

    # DEFINE CIRCLE CENTER #
    center = PointMixin()
    center.latitude = (extreme1.latitude + extreme2.latitude) / 2
    center.longitude = (extreme1.longitude + extreme2.longitude) / 2

    return extreme1, extreme2, center

def determine_direction(entry_lat, entry_lon, center_lat, center_lon):
    """
    Determines the direction the aircraft is coming from based on entry coordinates 
    relative to the circle center.

    Parameteres
    -------
    entry_lat : float
        entry point's longitude
    entry_lon : float
        entry point's longitude
    center_lat : float
        center coordinates' latitude
    center_lon : float
        center coordinates' longitude

    Returns
    -------
    dir : character
        A character with the direction of aircraft
    """
    delta_lat = entry_lat - center_lat  # Difference in latitude
    delta_lon = entry_lon - center_lon  # Difference in longitude

    dir = ''
    if delta_lat > 0:
        dir += 'N'
    else: 
        dir += 'S'
    
    if delta_lon > 0:
        dir += 'O'
    else:
        dir += 'W'
    
    return dir

def normalize_track(track: list, time: list) -> list:
    """
    Normalize the track data

    Parameters
    ----------
    track : list
        A list or array containing the track data
    time : list
        A list or array of timestamps corresponding to the track data
    window_size : int
        window size for smoothing the track

    Returns
    -------
    numpy array
        An array of the normalized track data
    """
    time = pd.to_datetime(time)

    ## CLEAN TRACK VALUES ##
    nan_indices = np.isnan(track)
    track_clean = np.array(track)[~nan_indices]  # Cleaned track
    
    if len(track_clean) == 0:
        return None

    ## UNWRAP CLEANED TRACK VALUES ##
    unwrapped_track = np.unwrap(np.radians(track_clean))
    track_clean = np.degrees(unwrapped_track)

    ## INTERPOLATE NaN VALUES with linear interpolation ##
    track_interpolated = np.interp(np.arange(len(track)), np.where(~nan_indices)[0], track_clean)

    ## Normalize track to [0, 1] ##
    track_min = np.min(track_interpolated)
    track_max = np.max(track_interpolated)
    
    # Check for division by zero or constant data (no variance) #
    if track_max == track_min:
        normalized_track = np.zeros_like(track_interpolated)
    else:
        normalized_track = (track_interpolated - track_min) / (track_max - track_min)
    
    return np.array(normalized_track)

def compute_horizontal_angle(x1_lat, x1_long, x2_lat, x2_long, longitudes, latitudes):
    """
    Computes the horizontal angle with respect to the midpoint of the runway's threshold using the angle between two vectors

    Parameters
    ----------
    x1_lat, x1_long : float
        Latitude and longitude of one point of the runway threshold.
    x2_lat, x2_long : float
        Latitude and longitude of the other point of the runway threshold.
    longitudes, latitudes : array-like
        Arrays of longitudes and latitudes of the trajectory points.

    Returns
    ----------
    angles : array-like
        Array of the horizontal angles in degrees.
    """
    # Calculate the midpoint of the runway threshold #
    mid_lat = (x1_lat + x2_lat) / 2
    mid_long = (x1_long + x2_long) / 2
    
    # Calculate vectors CP and AB #
    delta_x_CP = longitudes - mid_long
    delta_y_CP = latitudes - mid_lat
    delta_x_AB = x2_long - x1_long
    delta_y_AB = x2_lat - x1_lat
    
    # Calculate the dot product and magnitudes of the vectors #
    dot_product = delta_x_CP * delta_x_AB + delta_y_CP * delta_y_AB
    # cross_product = delta_x_CP * delta_y_AB - delta_y_CP * delta_x_AB #
    magnitude_CP = np.sqrt(delta_x_CP**2 + delta_y_CP**2)
    magnitude_AB = np.sqrt(delta_x_AB**2 + delta_y_AB**2)
    
    # Calculate the angle between the vectors in radians and convert to degrees #
    cos_theta = dot_product / (magnitude_CP * magnitude_AB)
    angle = np.rad2deg(np.arccos(cos_theta))

    return angle

def detect_trajectories(flight: Flight, maxtime: float = 5.0, debug: bool = False) -> Traffic:
    """
    Splits one flight into different segments based on the time difference between them (INPUT maxtime),
    e.g. if there is a difference between timestamps of more than maxtime, then there are different
    trajectories in this flight (because they ocurred at different moments in time)

    Parameters
    ----------
    flight : Flight
        Flight to be analyzed
    maxtime : float
        maximum difference of time
    debug : bool
        Debug toggle
 
    Returns
    ----------
    list
        list of cropped flights
    """
    trajectories = []
    lastIndex = 0   # initialize an index as the first position of the cropped flights
    id = 0          # id to assign with flight_id to cropped flights

    for i in range(1, len(flight.data)):
        time_diff = (flight.data.timestamp.iloc[i] - flight.data.timestamp.iloc[i-1]).total_seconds()

        if time_diff > maxtime:
            timeIni = flight.data.timestamp.iloc[lastIndex]
            timeLast = flight.data.timestamp.iloc[i]
            
            # for debug
            if debug:
                print(f"Initial Position Index: {lastIndex} at time: {timeIni}")
                print(f"Last Position Inex: {i} at time {timeLast} \n")
           
            flight_segment = flight.between(timeIni, timeLast) # cropped flight

            if flight_segment is not None: 
                id += 1 # add one to the id
                flight_segment = flight_segment.assign_id(idx=id)
                trajectories.append(flight_segment) # add cropped flight to variable trajectories
                lastIndex = i # update last index for initial position                 
            
    # at the end, add the final segment of the flight #
    timeIni = flight.data.timestamp.iloc[lastIndex]
    timeLast = flight.data.timestamp.iloc[len(flight.data)-1]

    if debug:
        print(f"LAST SEGMENT: Initial Position Index: {lastIndex} at time {timeIni}")
        print(f"LAST SEGMENT: Last Position Index: {len(flight.data)-1} at time {timeLast} \n")

    flight_segment = flight.between(timeIni, timeLast)
    if flight_segment is not None:    
        flight_segment = flight_segment.assign_id(idx=id+1)
        trajectories.append(flight_segment)            
    
    return trajectories

def process_flights_segments(flight: Flight, x1: list, x2: list, x3: list, x4: list, segments: list, debug: bool = False):
    """
    Processes a flight and splits it into segments defined by a list of ILS segments. For each resulting 
    sub-segment of the flight, the function extracts associated feature arrays (distance to airport, 
    horizontal angle, altitude, track) and stores the results along with timestamps and metadata

    The segmentation is based on the time intervals between consecutive ILS segment start times, dividing 
    the flight into pre-, inter-, and post-segment phases

    Parameters
    ----------
    flight : Flight
        Flight object to be analyzed. Must have a `.data` attribute with a timestamp column and a `.between(start, end)` method
    x1 : list or np.ndarray
        Feature array representing distance to the airport (in nautical miles)
    x2 : list or np.ndarray
        Feature array representing horizontal angle (in degrees)
    x3 : list or np.ndarray
        Feature array representing geo-altitude (in feet)
    x4 : list or np.ndarray
        Feature array representing normalized track (unitless)
    segments : list
        List of segment objects, each with a `.start` timestamp attribute indicating the beginning of the ILS segment
    debug : bool, optional
        Debug toggle

    Returns
    ----------
    np.ndarray
        A 2D object array of shape (num_valid_segments, 7), where each row contains:
        [0] - x1 slice for the segment (distance to airport)
        [1] - x2 slice for the segment (horizontal angle)
        [2] - x3 slice for the segment (geo-altitude)
        [3] - x4 slice for the segment (normalized track)
        [4] - pandas Series of timestamps for the segment
        [5] - Flight object cropped to the segment
        [6] - List [start_index - 1, end_index] indicating the position of the segment within the full flight
    """
    results = np.empty((len(segments) + 1, 7), dtype=object)
    for i in range(len(segments) + 1):

        if i == 0:
            start_time = flight.data.timestamp.iloc[0]
            end_time = segments[0].start         
            ikeep = 0
        elif i == len(segments):
            start_time = segments[i-1].start
            end_time = flight.data.timestamp.iloc[-1]
        else:
            start_time = segments[i-1].start
            end_time = segments[i].start

        # extract flight segment between detected segments #
        flight_segment = flight.between(start_time, end_time)    

        x1 = np.nan_to_num(x1, nan=0)  
        x2 = np.nan_to_num(x2, nan=0)  
        x3 = np.nan_to_num(x3, nan=0)  
        x4 = np.nan_to_num(x4, nan=0)  

        if flight_segment is None or flight_segment.data.empty or any(v is None for v in [x1, x2, x3, x4]):
            if debug: print(f"Skipping segment {i} due to None")
            continue
        
        #start_index = flight.data.index.get_loc(flight_segment.data.index[0])
        #end_index = flight.data.index.get_loc(flight_segment.data.index[-1]) + 1
        start_index = flight.data.index.get_indexer([flight_segment.data.index[0]])[0]
        end_index   = flight.data.index.get_indexer([flight_segment.data.index[-1]])[0] + 1

        if debug: print(f"FLIGHT SEGMENT {ikeep} | Start Index : {start_index} | End Index : {end_index} ")

        # Feature x1 - distance to airport (NM) #
        results[ikeep,0] = x1[start_index:end_index]
        
        # Feature x2 - horizontal angle (deg.) #
        results[ikeep,1] = x2[start_index:end_index]
        
        # Feature x3 - geoaltitude (ft.) #
        results[ikeep,2] = x3[start_index:end_index]
        
        # Feature x4 - normalized track (-) #
        results[ikeep,3] = x4[start_index:end_index]

        # Timestamps - traffic lib format #
        results[ikeep,4] = flight_segment.data.timestamp

        # Pack flight segment into results #
        results[ikeep,5] = flight_segment

        # Pack starting and ending index into results #
        results[ikeep,6] = [start_index-1, end_index]

        # use index ikeep instead of i, since you want to keep with only ikeep subset of i valid flight_segments #
        ikeep = ikeep + 1
    
    # if once the loop has finished, there are less valid flight_segments than at the beginning, shrink the initial results array #   
    if (len(segments) + 1) > ikeep:
        results = results[:ikeep,:]
    return results

def plot_flight_segments(results: np.ndarray, airport: str, num_segments: int, predictions: list):
    """
    Plots a comprehensive multi-panel visualization of segmented flight data, including time series 
    plots, geographic flight paths, and circuit classification results for each segment.

    The function creates a grid of subplots for each segment, displaying:
        - Distance to airport over time
        - Horizontal angle over time
        - Altitude over time
        - Normalized track over time
        - Geographic flight path with runway overlays
        - Predicted airdrome circuit label as a table

    Parameters
    ----------
    results : np.ndarray
        A structured array (as returned by `process_flights_segments`) containing per-segment data including:
        distance to airport, horizontal angle, geo-altitude, normalized track, timestamps, and the segment itself
    airport : str
        airport of interest for the given flight
    num_segments : int
        Number of flight segments to visualize (usually len(results))
    predictions : list
        List of predicted airdrome circuit labels (e.g., ["TRUE", "FALSE", ...]), one per segment

    Returns
    ----------
    None
        The function generates and displays an interactive plot using Plotly.
    """
    ## NOT SHOW LOGGING WARNINGS ##
    logging.getLogger().setLevel(logging.CRITICAL)
   
    ## INITIALIZE SUBPLOTS ##
    fig = make_subplots(
        rows = num_segments,    # number of rows
        cols = 6,               # number of columns
        subplot_titles=[title for i in range(num_segments + 1) for title in (f'Distance [NM] segment {i}', f'Horizontal angle [deg.] segment {i}',
        f'Local Altitude [ft.] segment {i}', f'Normalized Track [-] segment {i}', f'Flight path segment {i}', '')],
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "geo"}, {"type": "domain"}] for _ in range(num_segments)] # titles of subplots
    )

    for i in range(num_segments):      
        timestamps = results[i][4]
              
        ## DISTANCE TO AIRPORT PLOT (Subplot 1) ##
        fig.add_trace(go.Scatter(x=timestamps, y=results[i][0], 
        mode='lines', name=f'Distance to Airport (NM) Segment {i+1}', line=dict(color='blue')), row =i+1, col = 1)
        
        ## HORIZONTAL ANGLE PLOT (Subplot 2) ##
        fig.add_trace(go.Scatter(x=timestamps, y=results[i][1], 
        mode='lines', name=f'Horizontal Angle (deg.) Segment {i+1}', line=dict(color='orange')), row =i+1, col = 2)

        ## GEO ALTITUDE PLOT (Subplot 3) ##
        fig.add_trace(go.Scatter(x=timestamps, y=results[i][2], 
        mode='lines', name=f'Geo Altitude (ft.) Segment {i+1}', line=dict(color='magenta')), row =i+1, col = 3)

        ## NORMALIZED TRACK PLOT (Subplot 4) ##
        fig.add_trace(go.Scatter(x=timestamps, y=results[i][3], 
        mode='lines', name=f'Normalized Track (-) Segment {i+1}', line=dict(color='turquoise')), row =i+1, col = 4)
        
        ## MAP PLOT (Subplot 5)
        flight_segment = results[i][5]
        fig.add_trace(go.Scattergeo(
            lon=flight_segment.data.longitude,
            lat=flight_segment.data.latitude,
            mode='lines',
            line=dict(width=2, color='red'),
            name=f'Flight Path Segment {i+1}'
        ), row=i+1, col=5)

        ## Add runway locations to the map for each subplot
        runways = airport.runways
        for j in range(0, len(runways), 2):
            threshold_1_lon = runways[j].longitude
            threshold_1_lat = runways[j].latitude
            threshold_2_lon = runways[j+1].longitude
            threshold_2_lat = runways[j+1].latitude

            fig.add_trace(go.Scattergeo(
                lon=[threshold_1_lon, threshold_2_lon],
                lat=[threshold_1_lat, threshold_2_lat],
                mode='lines',
                line=dict(width=3, color='black'),
                name=f'Runway {runways.data.name.loc[j]}',
                opacity=0.75,
            ), row=i+1, col=5)

        ## Add annotation for AIRDROME CIRCUIT ##
        airdrome_circuit = predictions[i] #"TRUE" if predictions[i] == 1 else "FALSE"
        fig.add_trace(go.Table(
            header=dict(values=["AIRDROME CIRCUIT"]),
            cells=dict(values=[[airdrome_circuit]])
        ), row=i+1, col=6)

    ## Update layout for the map plots with more zoom
    for i in range(num_segments):
        flight_segment = results[i][5]
        fig.update_geos(
            projection_type="equirectangular",
            landcolor='rgb(243, 243, 243)',
            lonaxis_range=[min(flight_segment.data.longitude)-0.02, max(flight_segment.data.longitude)+0.02],
            lataxis_range=[min(flight_segment.data.latitude)-0.02, max(flight_segment.data.latitude)+0.02],
            row=i+1,
            col=5
        )

    ## FINAL OVERALL SETTING CONFIGURATION ##
    fig.update_layout(
        title = 'Flight Segments Analysis',
        showlegend = False,
        height = 300 * (num_segments),
        width = 1750,
    )

    fig.show()

def create_runway_area(airport: str, rwy: str, scale: float = 1.0, debug: bool = False) -> list:
    """
    Creates a rectangle area around an specified runway of an airport

    Parameters
    ----------
    airport : str
        ICAO identifier of the ariport
    rwy : str
        Runway identifier, e.g. '06'
    scale: float
        scale parameter to enlarge capture area in transversal direction of rwy
    debug: bool
        Debug toggle

    Returns
    ----------
    area: list
        List of coordinates representing the rotated bounding box
    """
    # convert the runway number to an angle
    if rwy.isdigit():  # numeric rwy (no side letter)
        num = int(rwy)
    else:
        num = int(rwy[:-1])

    # convert the runway number to an angle
    rwy_angle = int(num)*10
    if rwy_angle >= 180:
        rwy_angle = np.radians(rwy_angle - 180)
    else:
        rwy_angle = np.radians(rwy_angle)

    extreme1, extreme2, _ = retrieve_runway_information(airport, rwy)

    points = np.array([
        [extreme1.longitude - scale/1e3*np.sin(np.pi/2 - rwy_angle), extreme1.latitude + scale/5e3*np.cos(np.pi/2 - rwy_angle)], # south west (A)
        [extreme2.longitude - scale/1e3*np.sin(np.pi/2 - rwy_angle), extreme2.latitude + scale/5e3*np.cos(np.pi/2 - rwy_angle)], # south east (B)
        [extreme2.longitude + scale/1e3*np.sin(np.pi/2 - rwy_angle), extreme2.latitude - scale/5e3*np.cos(np.pi/2 - rwy_angle)], # north east (C)
        [extreme1.longitude + scale/1e3*np.sin(np.pi/2 - rwy_angle), extreme1.latitude - scale/5e3*np.cos(np.pi/2 - rwy_angle)]  # north west (D)
    ])

    # calculate the length of the runway aligned side
    Xlength = np.linalg.norm(points[1] - points[0])  # Length from A to B
    new_Xlength = Xlength * scale

    rwy_vecX = (points[1] - points[0]) / Xlength # (from A to B)

    displacX = new_Xlength/2 - (Xlength/2)

    # update points
    A = points[0] - rwy_vecX*displacX
    B = points[1] + rwy_vecX*displacX
    C = points[2] + rwy_vecX*displacX
    D = points[3] - rwy_vecX*displacX

    if debug:
        print(f"A: ({A[0]}, {A[1]})\nB: ({B[0]}, {B[1]})\nC: ({C[0]}, {C[1]})\nD: ({D[0]}, {D[1]})")

    area = [
        [A[1],A[0]],
        [B[1],B[0]],
        [C[1],C[0]],
        [D[1],D[0]],
        [A[1],A[0]],
    ]
    return area

def aligned_over_runway(flight: Flight, airport: str, rwy: str, scale: float = 1.0, debug: bool = False):
    """
    Creates and area around the runway and detects any trajectories inside that area, retrieving flight segments

    Parameters
    ----------
    flight : Flight
        Flight to be analyzed
    airport : str
        airport of interest for the given flight
    rwy : str
        Runway identifier, e.g. '06'
    scale : float
        parameter that magnifies the area of capture around the runway
    debug : bool
        Debug toggle

    Returns
    ----------
    list
        list of flights segments inside the created area around that specified runway
    """
    area = create_runway_area(airport, rwy, scale)

    # check if there are flight segments inside created areas
    west = min(area[0][1],area[3][1])
    south = min(area[0][0],area[1][0])
    east = max(area[1][1],area[2][1]) 
    north = max(area[2][0],area[3][0])

    if south > north: 
        oldsouth = south
        south = north
        north = oldsouth
    elif west > east:
        oldwest = west
        west = east
        east = oldwest
    
    segments = flight.inside_bbox([west,south,east,north]) # a tuple of floats (west, south, east, north)
    if segments is not None:
        return detect_trajectories(segments, debug=debug)

def detect_runway(flight: Flight, airport: str, scale: float = 1.0, max_scale: float = 10.0):
    """
    Given a flight and an airport, detects the most probable runway the flight used based on trajectory alignment.
    If no runway is found initially, the function automatically increases the capture area (scale) until a match is found
    or a maximum scale is reached.

    Parameters
    ----------
    flight : Flight
        Flight to be analyzed.
    airport : str
        ICAO or identifier of the airport of interest
    scale : float, optional
        Initial scaling factor that magnifies the area considered around each runway. Default is 1.0
    max_scale : float, optional
        Maximum scale limit to avoid infinite loop. Default is 10.0

    Returns
    ----------
    str
        Name of the most probable runway (e.g., "27L", "09R") that the flight used, based on data alignment
    -1
        If no valid runway was detected after scale increase
    """
    while scale <= max_scale:
        exit = False
        largest_seg = 0
        most_probable_rwy = None

        available_rwys = airport.runways.data.name
        for rwy in available_rwys:
            segments = aligned_over_runway(flight, airport, rwy, scale=scale)

            if segments is not None:
                len_seg = sum(len(segment.data) for segment in segments)

                if len_seg > largest_seg:
                    exit = True
                    most_probable_rwy = rwy
                    largest_seg = len_seg

        if exit:
            return most_probable_rwy, scale

        scale += 0.2

    print("No valid runway detected.")
    return -1, 0
    
def find_aerodrome_circuits(flight: Flight, airport: str = None, rwy: str = None, scale: float = 1.0, debug: bool = False, plot: bool = False):
    """
    Based on a trained Machine Learning model, it analyzes the flight and finds the number of detected airdrome circuits
    on the flight, as well as the time interval they ocurred

    Parameters
    ----------
    flight : Flight
        Flight to be analyzed
    airport : str = None
        ICAO code of the airport, if not provided it will just use the detected landing airport
    rwy : str
        Runway identifier, e.g. '06'
    scale : float = 1.0
        scale the area of capture so to enlarge it if necessary
    debug : bool = False
        debug toggle to print ILS segments information
    plot : bool = False
        plot debug toggle

    Returns
    ----------
    """
    ## FIND LANDING AIRPORT IF AIRPORT NOT GIVEN ##
    if airport == None: airport = flight.landing_airport()
    extreme1, extreme2, center = retrieve_runway_information(airport, rwy)
    
    ## LOAD MACHINE LEARNING MODEL ##
    model = joblib.load("ML/Models/RandomForest/advanced_model_lommis_rf.pkl")

    ## COMPUTE FEATURES ##
    x1 = np.array( flight.distance(center).data[["distance"]] ).flatten()
    x2 = np.array(compute_horizontal_angle(extreme1.latitude, extreme1.longitude, extreme2.latitude, extreme2.longitude, 
                                            flight.data.longitude.values, flight.data.latitude.values) )
    x3 = np.array(flight.data.altitude)
    x4 = normalize_track(flight.data.track, flight.data.timestamp)
    
    ## IDENTIFY ALIGNED OVER RUNWAY SEGMENTS ##
    flight_segments = aligned_over_runway(flight, airport, rwy, scale)

    ## PROCESS ALIGNED OVER RUNWAY SEGMENTS ##
    results = process_flights_segments(flight, x1, x2, x3, x4, flight_segments, debug)
    
    ## PREPARE FLIGHT SEGMENTS ##
    fixed_length = 500
    x_array = []
    for i, (x1_seg, x2_seg, x3_seg, x4_seg, timestamps, flight_segment, index) in enumerate(results):
        
        segment_length = len(flight_segment.data)
        if debug: print(f"INITIAL SEGMENT LENGTH {i} : {len(x1)}")

        # If segment is too long, downsample to fixed length
        if segment_length > fixed_length:
            step = int(np.floor(segment_length / fixed_length))
            x1_seg = x1_seg[::step][:fixed_length]
            x2_seg = x2_seg[::step][:fixed_length]
            x3_seg = x3_seg[::step][:fixed_length]
            x4_seg = x4_seg[::step][:fixed_length]
        
        # If segment is too short, upsample to fixed length
        elif segment_length < fixed_length:
            resampled_indices = np.linspace(0, segment_length - 1, fixed_length).astype(int)
            x1_seg = x1_seg[resampled_indices]
            x2_seg = x2_seg[resampled_indices]
            x3_seg = x3_seg[resampled_indices]
            x4_seg = x4_seg[resampled_indices]
        
        if debug: print(f"RESAMPLED SEGMENT LENGTH {i} : {len(x1)}")

        # Ensure all arrays are the same length
        x1_seg = np.pad(x1_seg, (0, fixed_length - len(x1_seg)), 'constant', constant_values=0)
        x2_seg = np.pad(x2_seg, (0, fixed_length - len(x2_seg)), 'constant', constant_values=0)
        x3_seg = np.pad(x3_seg, (0, fixed_length - len(x3_seg)), 'constant', constant_values=0)
        x4_seg = np.pad(x4_seg, (0, fixed_length - len(x4_seg)), 'constant', constant_values=0)

        # Initialize segment data array
        segment_data = np.zeros((fixed_length, 4))

        # Populate segment data
        segment_data[:,0] = x1_seg
        segment_data[:,1] = x2_seg
        segment_data[:,2] = x3_seg
        segment_data[:,3] = x4_seg

        # Append processed segment data to x array
        x_array.append(segment_data)
    
    # Convert x_array to a numpy array with the correct shape for the ML algorithm #
    x_array = np.array(x_array)  # Desired Shape: [number of samples, fixed_length, num_features]

    ## PREDICT USING THE IMPORTED MODEL ##
    x_array = x_array.reshape(x_array.shape[0], -1)
    y_pred = model.predict(x_array)
    y_pred = np.array(y_pred).flatten()
    y_predbin = (y_pred > 0.5).astype(int)

    ## OPTIONAL OUTPUT : PLOT FLIGHT SEGMENTS ##
    if plot:
        plot_flight_segments(results, airport, len(results[:,]), y_pred)

    if debug: print(y_predbin)
    
    ## OUTPUT: NUMBER OF CIRCUITS ##
    ncircuits = len(y_predbin[y_predbin == 1])
    if ncircuits >=1:
        indexes = [(item[0], item[1]) for item, pred in zip(results[:, 6], y_predbin) if pred == 1]
        if debug: print(f"Number of airdrome circuits detected: {ncircuits}")
        return ncircuits, indexes
    else:
        if debug: print("No airdrome circuit detected")
        return 0, None
    
def get_aircraft_type(registration_number: str, method: int = 1, debug: bool = False):
    """
    Retrieves the mapped aircraft type for a given registration number using a predefined classification scheme.

    The function reads from a Swiss civil aircraft registry CSV file (`register_BAZL_aircraft.csv`) and matches 
    the provided registration number. It then maps the detailed aircraft type (Luftfahrzeugtyp) to a simplified 
    category based on a predefined code

    Parameters
    ----------
    registration_number : str
        The aircraft registration number (e.g., "HBXYZ" or "HB-XYZ")
    method : int, optional
        Determines the formatting of the registration number before lookup:
        - 1: use raw format (e.g., "HBXYZ")
        - 2: insert dash after country code (e.g., "HB-XYZ")
        Default is 1
    debug : bool, optional
        Debug toggle

    Returns
    ----------
    str or None
        Mapped aircraft type code:
            "A" - Motorflugzeug
            "B" - Helicopter
            "C" - Powered Glider
            "D" - Glider
            "E" - Ecolight / Ultralight
            "F" - Balloons / Airships
            "G" - Gyrocopters / Other VTOL
        Returns `None` if no match is found in the registry
    """
    # Define the mapping for Luftfahrzeugtyp
    mapping = {
        "Aeroplane": "A",  # Motorflugzeug
        "Airship (Gas)": "F",  # Luftschiffe
        "Airship (Hot-air)": "F",  # Luftschiffe
        "Balloon (Gas)": "F",  # Luftschiffe
        "Balloon (Hot-air)": "F",  # Luftschiffe
        "Ecolight": "E",  # Ecolight
        "Glider": "D",  # Segelflugzeug (inkl. Selbststart)
        "Gyrocopter": "G",  # andere VTOL als Helikopter (z.B. Gyrocopter, Tilt Rotor)
        "Helicopter": "B",  # Helikopter
        "Homebuild Glider": "D",  # Segelflugzeug (inkl. Selbststart)
        "Homebuilt Airplane": "A",  # Motorflugzeug
        "Homebuilt Gyrocopter": "G",  # andere VTOL als Helikopter (z.B. Gyrocopter, Tilt Rotor)
        "Homebuilt Helicopter": "B",  # Helikopter
        "Powered Glider": "C",  # Motorsegler (TMG)
        "RPAS Aircraft": "A",  # Motorflugzeug
        "RPAS Rotorcraft": "B",  # Helikopter
        "RPAS Tethered Balloon (Gas)": "F",  # Luftschiffe
        "Tethered Balloon (Gas)": "F",  # Luftschiffe
        "Trike": "E",  # Ecolight
        "Ultralight (3-axis control)": "E",  # Ecolight
        "Ultralight Gyrocopter": "G"  # andere VTOL als Helikopter (z.B. Gyrocopter, Tilt Rotor)
    }

    # Function to map Luftfahrzeugtyp to the specified categories
    def map_aircraft(aircraft_type):
        for key in mapping:
            if key in aircraft_type:
                return mapping[key]
        return aircraft_type

    # Function to format the registration number
    def format_registration_number(reg_number):
        if len(reg_number) == 5:
            if method == 1:
                return reg_number
            else:
                return reg_number[:2] + '-' + reg_number[2:]
        return reg_number

    # Format the registration number
    formatted_registration_number = format_registration_number(registration_number)

    # Read the CSV file and process it
    with open('Excel/template/register_BAZL_aircraft.csv', 'r', encoding='utf-16') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        
        # Process each row
        for row in reader:
            if row[' Registration'].strip() == formatted_registration_number:
                aircraft_type = row[' Luftfahrzeugtyp']
                mapped_value = map_aircraft(aircraft_type)
                if debug: print(f"LfrID: {row['LfrID']}, Registration: {row[' Registration']}, Luftfahrzeugtyp: {mapped_value}")
                return mapped_value
    return None

def analyze_flight(flight: Flight, airport: str, flight_data: list, date: list, scale: float = 1.0, debug: bool = False):
    """
    Analyzes a flight in relation to a given airport and determines departure, landing, and circuit (volte) operations.
    It classifies the movement type, estimates runway and route direction, and appends structured results to `flight_data`

    The function performs:
        - Runway detection and metadata extraction
        - Aircraft type identification based on registration
        - Departure detection (type "D") and route direction inference
        - Circuit detection (type "V") including number of circuits and direction
        - Landing detection (type "A") and route direction inference

    ParametersW
    ----------
    flight : Flight
        Flight object containing the trajectory and metadata to analyze
    airport : str
        Airport  of interest (e.g., LSZT), used to detect runway and proximity
    flight_data : list
        External list where the extracted flight event data will be appended
    date : list of int
        Date of the flight in the format [YYYY, MM, DD]
    scale : float, optional
        Initial scale factor used for runway detection. Will increase progressively if needed. Default is 1.0
    debug : bool, optional
        Debug toggle

    Returns
    ----------
    None
        The function modifies the `flight_data` list in-place by appending rows representing:
        - Departures: ["LSZT", year, month, day, time, "D", ..., route]
        - Circuits:   ["LSZT", year, month, day, time, "V", ..., route]
        - Arrivals:   ["LSZT", year, month, day, time, "A", ..., route]
    """
    # EXTRACT RUNWAY INFORMATION #
    rwy, rwy_scale = detect_runway(flight, airport, scale, 10*scale)
    extreme1, extreme2, center = retrieve_runway_information(airport, rwy)

    # DEFINE RADIUS #
    upper_coords = (47.562333, 8.986976) 
    radius = geodesic((center.latitude, center.longitude), upper_coords)

    if flight.aircraft is None:
        aircraft_type = get_aircraft_type(flight.data.callsign.iloc[0], method = 2, debug=debug)
    else:
        aircraft_type = get_aircraft_type(flight.aircraft.registration, method = 1, debug=debug)
    
    landing_lszt = flight.landing_at(airport)      # landing bool
    takeoff_lszt = flight.takeoff_from(airport)    # takeoff bool

    # DEPARTURE IDENTIFICATION (D) #
    if takeoff_lszt:
        climb_segment = flight.query(f"vertical_rate > {flight.data.vertical_rate.iloc[0]}")
        if climb_segment is not None and not climb_segment.data.empty:
            starting_flight = climb_segment.first('1 min')

            d_24 = starting_flight.closest_point(extreme1)
            d_06 = starting_flight.closest_point(extreme2)

            if d_06.timestamp > d_24.timestamp:
                rwy_takeoff = '24'
                
                departure_time = d_06.timestamp
            else:
                rwy_takeoff = '06'

                departure_time = d_24.timestamp

            # Compute route_takeoff using entry direction detection
            dist = flight.distance(center)
            small_dist = dist.data[dist.data.distance.between(radius.nm - 0.1, radius.nm + 0.1)]
            small_dist.sort_values(by="timestamp")

            if len(small_dist) > 0:
                small_dist = small_dist.copy()
                small_dist["time_diff"] = small_dist["timestamp"].diff().dt.total_seconds()
                small_dist["crossing_id"] = (small_dist["time_diff"] > 120).cumsum()

                crossing_events = small_dist.groupby("crossing_id").agg(
                    entry_timestamp=("timestamp", "first"),
                    entry_latitude=("latitude", "first"),
                    entry_longitude=("longitude", "first"),
                ).reset_index(drop=True)

                # Determine route_takeoff direction
                if len(crossing_events) > 0:
                    route_takeoff = determine_direction(
                        crossing_events.iloc[0]["entry_latitude"], 
                        crossing_events.iloc[0]["entry_longitude"], 
                        center.latitude, 
                        center.longitude)
            else:
                route_takeoff = determine_direction(
                    starting_flight.data.latitude.iloc[-1],
                    starting_flight.data.longitude.iloc[-1],
                    center.latitude, 
                    center.longitude
                )

            if landing_lszt:
                ori_dest = "LSZT"            
            else:
                ori_dest = "LSZZ"
            
            flight_data.append(
                ["LSZT", date[0], date[1], date[2], departure_time.strftime("%H%M"), "D", 
                4, aircraft_type, flight.data.callsign.iloc[0], rwy_takeoff, 1, 0, 0, ori_dest, route_takeoff] 
            )

            if debug:
                print("------------------------------------------------")
                print(f"START FLIGHT: {flight.data.timestamp.iloc[0]}")
                print(f"DEPARTURE: {departure_time}")
                print(f"RUNWAY: {rwy_takeoff}")
                print(f"ROUTE: {route_takeoff}")
                print("------------------------------------------------")

    # VOLTE IDENTIFICATION (V) #
    ncircuits, indexes = find_aerodrome_circuits(flight, airport, rwy, rwy_scale, debug=debug, plot=False) 
    if ncircuits > 0:
        if takeoff_lszt:
            origin_dest = "LSZT"
        else:
            if landing_lszt:
                origin_dest = "LSZT"
            else:
                origin_dest = "LSZZ"
        
        start_volte = flight.data.timestamp.iloc[indexes[0][0]]
        end_volte = flight.data.timestamp.iloc[indexes[0][1]]
        volte_flight = flight.between(start_volte, end_volte)

        if volte_flight is not None:
            d_24 = volte_flight.closest_point(extreme1)
            d_06 = volte_flight.closest_point(extreme2)

            if d_06.timestamp > d_24.timestamp:
                rwy_volte = '24'
                route_volte = 'NO'
            else:
                rwy_volte = '06'
                route_volte = 'SW'

            flight_data.append(
                ["LSZT", date[0], date[1], date[2], start_volte.strftime("%H%M"), "V", 
                4, aircraft_type, flight.data.callsign.iloc[0], rwy_volte, ncircuits*2, 0, 0, origin_dest, route_volte]
            )

    # ARRIVAL IDENTIFICATION (A) #
    if landing_lszt:
        last_change_index = flight.data.iloc[::-1].vertical_rate.ne(flight.data.vertical_rate.iloc[-1]).idxmax()
        ending_flight = flight.query(f"index <= {last_change_index}").last('1 min')

        d_24 = ending_flight.closest_point(extreme1)
        d_06 = ending_flight.closest_point(extreme2)    

        if d_06.timestamp < d_24.timestamp:
            rwy_landing = '06'

            landing_time = d_06.timestamp
        else:
            rwy_landing = '24'

            landing_time = d_24.timestamp
    
        # Compute route_landing using entry direction detection
        dist = flight.distance(center)
        small_dist = dist.data[dist.data.distance.between(radius.nm - 0.1, radius.nm + 0.1)]
        small_dist.sort_values(by="timestamp")

        if len(small_dist) > 0:
            small_dist = small_dist.copy()
            small_dist["time_diff"] = small_dist["timestamp"].diff().dt.total_seconds()
            small_dist["crossing_id"] = (small_dist["time_diff"] > 120).cumsum()

            crossing_events = small_dist.groupby("crossing_id").agg(
                entry_timestamp=("timestamp", "first"),
                entry_latitude=("latitude", "first"),
                entry_longitude=("longitude", "first"),
            ).reset_index(drop=True)

            # Determine route_landing direction
            if len(crossing_events) > 0:
                if takeoff_lszt and len(crossing_events) > 1:
                    pos_landing = 1
                else: pos_landing = 0

            route_landing = determine_direction(
                crossing_events.iloc[pos_landing]["entry_latitude"], 
                crossing_events.iloc[pos_landing]["entry_longitude"], 
                center.latitude, 
                center.longitude
            )
        else:
            route_landing = determine_direction(
                ending_flight.data.latitude.iloc[-1],
                ending_flight.data.longitude.iloc[-1],
                center.latitude, 
                center.longitude
            )
        
        if takeoff_lszt:
            ori_dest = "LSZT"
        else:
            ori_dest = "LSZZ"

        flight_data.append(
            ["LSZT", date[0], date[1], date[2], landing_time.strftime("%H%M"), "A", 
                4, aircraft_type, flight.data.callsign.iloc[0], rwy_landing, 1, 0, 0, ori_dest, route_landing] 
        )

        if debug:
            print("------------------------------------------------")
            print(f"LANDING: {landing_time}")
            print(f"END FLIGHT: {flight.data.timestamp.iloc[-1]}")
            print(f"RUNWAY: {rwy_landing}")
            print(f"ROUTE: {route_landing}")
            print("------------------------------------------------")
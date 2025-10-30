import csv
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# traffc
import traffic
from traffic.core import Traffic, Flight
from traffic.data.basic.airports import Airport
from traffic.core.mixins import PointBase 
from traffic.data import airports

# for format
from typing import Union, Optional, Any, Dict, List

# for plotting
from pyproj import Transformer
import plotly.graph_objects as go
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
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
        opposite_num = (num + 18) % 36
        if opposite_num == 00: opposite_num = 36  # Handle wraparound
        return f"{opposite_num:02}"
        
    else:  # rwy with a side letter
        num, side = int(rwy[:-1]), rwy[-1]
        opposite_num = (num + 18) % 36
        if opposite_num == 00: opposite_num = 36  # in case it's 0, turn into 36 because 00 does not exist!
        opposite_num = f"{opposite_num:02}"
        opposite_side = 'R' if side == 'L' else 'L' if side == 'R' else side
        return f"{opposite_num}{opposite_side}"

def retrieve_runway_information(airport: Union[str, "Airport"], rwy: str):
    """
    Retrieves the geographic coordinates of both thresholds of a given runway, as well as the midpoint between them.

    This function calculates the geographic coordinates of the specified runway threshold (`extreme1`), the opposite threshold (`extreme2`), and the midpoint (`center`) between them. It uses the runway identifier (e.g., "06", "24") and the airport object to fetch the required data.

    Parameters
    ----------
    airport : str or Airport
        ICAO code of the airport (e.g., "EHAM", "KJFK").
    rwy : str
        Runway identifier (e.g., "06", "24", "18R").

    Returns
    ----------
    tuple
        A tuple containing three `Point` objects:
            - extreme1: Geographic coordinates of the specified runway threshold.
            - extreme2: Geographic coordinates of the opposite runway threshold.
            - center: Geographic coordinates of the midpoint between the two thresholds.
    """
    # Calculate names for lookup
    rwy_opposite = opposite_runway(rwy)

    airport = airport if isinstance(airport, Airport) else airports[airport]

    if airport.icao == "LSZF":
        extreme1_info = {'latitude': 47.443272, 'longitude': 8.230352, 'name': '08'}
        extreme2_info = {'latitude': 47.44196, 'longitude': 8.238349, 'name': '26'}
    else:
        # Logic to fetch runway data
        df = airport.runways.data
        
        # Filter for the extreme points
        extreme1_info = df[df.name == rwy].iloc[0]
        extreme2_info = df[df.name == rwy_opposite].iloc[0]

    # New way: Instantiating Point objects
    extreme1 = PointBase(
        latitude=extreme1_info['latitude'],
        longitude=extreme1_info['longitude'],
        altitude=airport.altitude,
        name=rwy
    )
    extreme2 = PointBase(
        latitude=extreme2_info['latitude'],
        longitude=extreme2_info['longitude'],
        altitude=airport.altitude,
        name=rwy_opposite
    )

    # DEFINE CENTER POINT #
    center_lat = (extreme1.latitude + extreme2.latitude) / 2
    center_lon = (extreme1.longitude + extreme2.longitude) / 2
    
    center = PointBase(
        latitude=center_lat,
        longitude=center_lon,
        altitude=airport.altitude,
        name="Center"
    )

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

def unwrap_track(track: list, time: list) -> list:
    """
    Unwraps the track data, cleaning and interpolating NaN values

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
        An array of the unwrapped track data
    """
    time = pd.to_datetime(time)

    ## CLEAN TRACK VALUES ##
    nan_indices = np.isnan(track)
    track_clean = np.array(track)[~nan_indices]
    
    if len(track_clean) == 0:
        return None

    ## UNWRAP CLEANED TRACK VALUES ##
    unwrapped_track = np.unwrap(np.radians(track_clean))
    track_clean = np.degrees(unwrapped_track)

    ## INTERPOLATE NaN VALUES with linear interpolation ##
    track_interpolated = np.interp(np.arange(len(track)), np.where(~nan_indices)[0], track_clean)
    
    return np.array(track_interpolated)

def compute_alignment_angle(x1_lat, x1_long, x2_lat, x2_long, longitudes, latitudes):
    """
    Computes the longitudinal alignment angle with respect to the midpoint of the runway's threshold 
    using the angle between two vectors

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
        Array of the alignment angles in degrees.
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

def detect_trajectories(flight: Flight, maxtime: float = 5.0, debug: bool = False) -> list:
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
    timestamps = flight.data.timestamp.values  # numpy array for speed
    diffs = np.diff(timestamps).astype("timedelta64[s]").astype(float)

    # split indices where gap > maxtime
    split_indices = np.where(diffs > maxtime)[0] + 1  

    trajectories = []
    start_idx = 0
    seg_id = 1

    for idx in split_indices:
        timeIni, timeLast = timestamps[start_idx], timestamps[idx]
        if debug:
            print(f"Segment {seg_id}: {timeIni} → {timeLast}")
        timeIni = pd.Timestamp(timeIni)
        timeLast = pd.Timestamp(timeLast)
        
        if timeIni.tzinfo is None:
            timeIni = timeIni.tz_localize('UTC')
        else:
            timeIni = timeIni.tz_convert('UTC')
        if timeLast.tzinfo is None:
            timeLast = timeLast.tz_localize('UTC')
        else:
            timeLast = timeLast.tz_convert('UTC')

        seg = flight.between(timeIni, timeLast)
        if seg is not None:
            trajectories.append(seg.assign_id(idx=seg_id))
        seg_id += 1
        start_idx = idx

    # add last segment
    timeIni = pd.Timestamp(timestamps[start_idx])
    timeLast = pd.Timestamp(timestamps[-1])
    if timeIni.tzinfo is None:
        timeIni = timeIni.tz_localize('UTC')
    else:
        timeIni = timeIni.tz_convert('UTC')
    if timeLast.tzinfo is None:
        timeLast = timeLast.tz_localize('UTC')
    else:
        timeLast = timeLast.tz_convert('UTC')
    
    seg = flight.between(timeIni, timeLast)
    if seg is not None:
        trajectories.append(seg.assign_id(idx=seg_id))

    return trajectories

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
    rwy_angle = np.radians((num * 10) % 180)

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

    lats, lons = zip(*area[:4])  
    south, north = min(lats), max(lats)
    west, east = min(lons), max(lons)
    
    segments = flight.inside_bbox([west,south,east,north]) # a tuple of floats (west, south, east, north)
    if segments is not None:
        return detect_trajectories(segments, debug=debug)

def detect_runway(flight: Flight, airport: Union[str, "Airport"], scale: float = 1.0, max_scale: float = 10.0):
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
    airport = airport if isinstance(airport, Airport) else airports[airport]
    
    available_rwys = airport.runways.data.name

    while scale <= max_scale:
        best_rwy, largest_seg = None, 0
        for rwy in available_rwys:
            segments = aligned_over_runway(flight, airport, rwy, scale=scale)
            if segments:
                total_len = sum(len(seg.data) for seg in segments)
                if total_len > largest_seg:
                    largest_seg, best_rwy = total_len, rwy
        if best_rwy:
            return best_rwy, scale
        scale += 0.2
    
    #print("No valid runway detected.")
    return -1, 0

def process_flights_segments(flight: Flight, x1: list, x2: list, x3: list, x4: list, segments: list, debug: bool = False):
    """
    Processes a flight and splits it into segments defined by a list of segments. For each resulting 
    sub-segment of the flight, the function extracts associated feature arrays (distance to airport, 
    alignment angle, altitude, track) and stores the results along with timestamps and metadata

    The segmentation is based on the time intervals between consecutive segment start times, dividing 
    the flight into pre-, inter-, and post-segment phases

    Parameters
    ----------
    flight : Flight
        Flight object to be analyzed. Must have a `.data` attribute with a timestamp column and a `.between(start, end)` method
    x1 : list or np.ndarray
        Feature array representing distance to the airport (in nautical miles)
    x2 : list or np.ndarray
        Feature array representing alingment angle (in degrees)
    x3 : list or np.ndarray
        Feature array representing geo-altitude (in feet)
    x4 : list or np.ndarray
        Feature array representing unwrapped track (unitless)
    segments : list
        List of segment objects, each with a `.start` timestamp attribute indicating the beginning of the segment
    debug : bool, optional
        Debug toggle

    Returns
    ----------
    np.ndarray
        A 2D object array of shape (num_valid_segments, 7), where each row contains:
        [0] - x1 slice for the segment (distance to airport)
        [1] - x2 slice for the segment (alignment angle)
        [2] - x3 slice for the segment (geo-altitude)
        [3] - x4 slice for the segment (unwrapped track)
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
        # Resolve start/end positions of the segment within the full flight data
        try:
            start_index = flight.data.index.get_indexer([flight_segment.data.index[0]])[0]
            end_index   = flight.data.index.get_indexer([flight_segment.data.index[-1]])[0] + 1
        except Exception as e:
            if debug:
                print(f"Skipping segment {i}: could not resolve start/end index - {e}")
            continue

        # Basic validation of resolved indices
        if start_index < 0 or end_index <= 0 or end_index <= start_index:
            if debug:
                print(f"Skipping segment {i}: invalid indices start={start_index}, end={end_index}")
            continue

        if start_index >= len(x1) or end_index > len(x1):
            if debug:
                print(f"Skipping segment {i}: indices out of range for feature arrays (start={start_index}, end={end_index}, len={len(x1)})")
            continue

        if flight_segment.data is None or flight_segment.data.empty:
            if debug:
                print(f"Skipping segment {i}: flight_segment has no data")
            continue

        if debug: print(f"FLIGHT SEGMENT {ikeep} | Start Index : {start_index} | End Index : {end_index} ")

        # Feature x1 - distance to airport (NM) #
        results[ikeep,0] = x1[start_index:end_index]
        
        # Feature x2 - alignment angle (deg.) #
        results[ikeep,1] = x2[start_index:end_index]
        
        # Feature x3 - geoaltitude (ft.) #
        results[ikeep,2] = x3[start_index:end_index]

        # Feature x4 - unwrapped track (-) #
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
        - Alignment angle over time
        - Altitude over time
        - Unwrapped track over time
        - Geographic flight path with runway overlays
        - Predicted airdrome circuit label as a table

    Parameters
    ----------
    results : np.ndarray
        A structured array (as returned by `process_flights_segments`) containing per-segment data including:
        distance to airport, alignment angle, geo-altitude, unwrapped track, timestamps, and the segment itself
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
        subplot_titles=[title for i in range(num_segments + 1) for title in (f'Distance [NM] segment {i}', f'Alignment angle [deg.] segment {i}',
        f'Local Altitude [ft.] segment {i}', f'Unwrapped Track [-] segment {i}', f'Flight path segment {i}', '')],
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "geo"}, {"type": "domain"}] for _ in range(num_segments)] # titles of subplots
    )

    phase_colors = {
        'CLIMB': 'rgba(192,255,192,0.5)',
        'DESCENT': 'rgba(255,214,214,0.5)',
        'LEVEL': 'rgba(255,240,122,0.5)',
        'NA': 'rgba(224,224,224,0.5)',
    }
    
    for i in range(num_segments):      
        # Normalize timestamps to millisecond precision to avoid Plotly nanosecond warnings
        timestamps = pd.to_datetime(results[i][4])
        try:
            timestamps = timestamps.dt.round('ms')
        except Exception:
            pass
        flight_segment = results[i][5]
        # Get resampled phases if available (from tuple index 7), otherwise fall back to original
        phases = results[i][7] if len(results[i]) > 7 else (flight_segment.data['phase'].values if 'phase' in flight_segment.data.columns else None)

        # Helper: add vertical background rectangles spanning full y for each contiguous phase block
        def add_phase_vrects(col_idx: int):
            if phases is None or len(phases) == 0:
                return
            # Ensure timestamps is a numpy/pandas indexable sequence of datetimes
            ts = timestamps
            # Find contiguous runs
            start = 0
            current = phases[0]
            for k in range(1, len(phases) + 1):
                if k == len(phases) or phases[k] != current:
                    x0 = ts.iloc[start] if hasattr(ts, 'iloc') else ts[start]
                    x1 = ts.iloc[k - 1] if hasattr(ts, 'iloc') else ts[k - 1]
                    color = phase_colors.get(str(current), 'rgba(224,224,224,0.5)')
                    # Add a vrect covering the whole subplot height
                    fig.add_vrect(
                        x0=x0,
                        x1=x1,
                        fillcolor=color,
                        opacity=0.8,
                        line_width=0,
                        layer='below',
                        row=i + 1,
                        col=col_idx,
                        exclude_empty_subplots=False,
                    )
                    # Start new run if not at end
                    if k < len(phases):
                        start = k
                        current = phases[k]

        # Add background rectangles to the four XY subplots
        add_phase_vrects(1)
        add_phase_vrects(2)
        add_phase_vrects(3)
        add_phase_vrects(4)

        ## DISTANCE TO AIRPORT PLOT (Subplot 1) ##
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=results[i][0],
                mode='lines',
                name=f'Distance to Airport (NM) Segment {i+1}',
                line=dict(color='blue'),
                showlegend=False,
            ),
            row=i + 1,
            col=1,
        )

        ## ALIGNMENT ANGLE PLOT (Subplot 2) ##
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=results[i][1],
                mode='lines',
                name=f'Alignment Angle (deg.) Segment {i+1}',
                line=dict(color='orange'),
                showlegend=False,
            ),
            row=i + 1,
            col=2,
        )

        ## GEO ALTITUDE PLOT (Subplot 3) ##
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=results[i][2],
                mode='lines',
                name=f'Geo Altitude (ft.) Segment {i+1}',
                line=dict(color='magenta'),
                showlegend=False,
            ),
            row=i + 1,
            col=3,
        )

        ## UNWRAPPED TRACK PLOT (Subplot 4) ##
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=results[i][3],
                mode='lines',
                name=f'Unwrapped Track (-) Segment {i+1}',
                line=dict(color='turquoise'),
                showlegend=False,
            ),
            row=i + 1,
            col=4,
        )

        ## MAP PLOT (Subplot 5)
        fig.add_trace(go.Scattergeo(
            lon=flight_segment.data.longitude,
            lat=flight_segment.data.latitude,
            mode='lines',
            line=dict(width=2, color='red'),
            name=f'Flight Path Segment {i+1}',
            showlegend=False,
        ), row=i+1, col=5)

        ## Add runway locations to the map for each subplot
        runway_data = airport.runways.data
        for j in range(0, len(runway_data), 2):
            threshold_1_lon = runway_data.iloc[j].longitude
            threshold_1_lat = runway_data.iloc[j].latitude
            threshold_2_lon = runway_data.iloc[j+1].longitude
            threshold_2_lat = runway_data.iloc[j+1].latitude

            fig.add_trace(go.Scattergeo(
                lon=[threshold_1_lon, threshold_2_lon],
                lat=[threshold_1_lat, threshold_2_lat],
                mode='lines',
                line=dict(width=3, color='black'),
                name=f'Runway {runway_data.name.iloc[j]}',
                opacity=0.75,
                showlegend=False,
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

    # Add one-time legend entries for phases (shapes don't appear in legend)
    # Add after plotting real data so axis type inference is not affected.
    for _phase_name, _phase_color in phase_colors.items():
        fig.add_trace(
            go.Scatter(
                x=[],  # empty data so it doesn't interfere with axes/zoom
                y=[],
                mode='markers',
                marker=dict(size=10, color=_phase_color, symbol='square'),
                name=_phase_name,
                showlegend=True,
                hoverinfo='skip',
            ),
            row=1,
            col=1,
        )

    ## FINAL OVERALL SETTING CONFIGURATION ##
    fig.update_layout(
        title = 'Flight Segments Analysis',
        showlegend = True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
        ),
        height = 300 * (num_segments),
        width = 1750,
    )

    fig.show()

def _load_model_and_config(model_name: str, debug: bool = False):
    """
    Helper function to load model, scaler, and determine feature requirements.
    
    Parameters
    ----------
    model_name : str
        One of: 'rf', 'lr', 'lstm', 'blstm', 'cnn'
    debug : bool
        Debug toggle
    
    Returns
    -------
    tuple : (model, scaler, requires_phases, requires_scaling, is_deep_learning)
    """
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    
    model_paths = {
        'rf': 'ML/Models/RandomForest/25102025_rf_engineered.pkl',
        'lr': 'ML/Models/LogisticRegression/25102025_logreg_engineered.pkl',
        'lstm': 'ML/Models/LSTM/LSTM_24102025_circuits.keras',
        'blstm': 'ML/Models/LSTM/LSTM_improved_bidirectional.keras',
        'cnn': 'ML/Models/CNN_1D/CNN_1D_24102025_circuits.h5',
    }
    
    scaler_path = Path("ML/Models/scaler_fixed.pkl")
    
    # Load model
    model_path = Path(model_paths.get(model_name.lower()))
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Please train the model first.")
    
    if model_name.lower() in ['lstm', 'blstm', 'cnn']:
        # Deep learning models
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.losses import BinaryCrossentropy
        
        model = tf.keras.models.load_model(str(model_path), compile=False)
        model.compile(optimizer=Adam(learning_rate=5e-4), loss=BinaryCrossentropy(from_logits=True))
        
        # Load scaler for numeric features
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(f"Scaler not found: {scaler_path}. Please run training notebook cell 18 to create it.")
        
        if debug: print(f"Loaded {model_name.upper()} model (deep learning, requires phases + scaling)")
        return model, scaler, True, True, True  # requires_phases, requires_scaling, is_deep_learning
    
    elif model_name.lower() in ['rf', 'lr']:
        # Sklearn models with engineered features (no phases, no scaling needed - pipeline handles it)
        model = joblib.load(model_path)
        
        if debug: print(f"Loaded {model_name.upper()} model (engineered features, pipeline handles scaling)")
        return model, None, False, False, False  # no phases, no separate scaling, not deep learning
    
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from: 'rf', 'lr', 'lstm', 'blstm', 'cnn'")

def find_aerodrome_circuits(flight: Flight, airport: str = None, rwy: str = None, model: str = 'rf', scale: float = 1.0, sigma: float = 1.0, debug: bool = False, plot: bool = False):
    """
    Based on a trained Machine Learning model, analyses the flight and finds the number of detected airdrome circuits
    on the flight, as well as the time interval they occurred.

    Parameters
    ----------
    flight : Flight
        Flight to be analyzed
    airport : str = None
        ICAO code of the airport, if not provided it will use the detected landing airport
    rwy : str
        Runway identifier, e.g. '06'
    scale : float = 1.0
        Scale the area of capture to enlarge it if necessary
    sigma : float = 1.0
        Gaussian filter sigma for smoothing features
    debug : bool = False
        Debug toggle to print segments information
    plot : bool = False
        Plot debug toggle
    model : str = 'rf'
        Model type to use. Options: 'rf', 'lr', 'lstm', 'blstm', 'cnn'
        - 'rf': Random Forest (engineered features)
        - 'lr': Logistic Regression (engineered features)
        - 'lstm': LSTM (sequential, with phases)
        - 'blstm': Bidirectional LSTM (sequential, with phases)
        - 'cnn': 1D CNN (sequential, with phases)

    Returns
    -------
    tuple : (results, y_predbin, n_circuits, indexes)
        - results: numpy array of segment analysis results
        - y_predbin: binary predictions for each segment
        - n_circuits: total number of circuits detected
        - indexes: time indexes of detected circuits
    """
    ## FIND LANDING AIRPORT IF AIRPORT NOT GIVEN ##
    if airport == None: airport = flight.landing_airport()
    airport_obj = airport if isinstance(airport, Airport) else airports[airport]
    extreme1, extreme2, center = retrieve_runway_information(airport_obj, rwy)
    
    ## LOAD MODEL AND CONFIGURATION ##
    model_obj, scaler, requires_phases, requires_scaling, is_deep_learning = _load_model_and_config(model, debug)

    ## COMPUTE FEATURES ##
    flight = flight.phases()
    
    x1 = gaussian_filter1d(np.array( flight.distance(center).data[["distance"]] ).flatten(), sigma=sigma)
    x2 = gaussian_filter1d(np.array(compute_alignment_angle(extreme1.latitude, extreme1.longitude, extreme2.latitude, extreme2.longitude, 
                                            flight.data.longitude.values, flight.data.latitude.values)), sigma=sigma)
    x3 = gaussian_filter1d(np.array(flight.data.altitude), sigma=sigma)
    x4 = gaussian_filter1d(unwrap_track(flight.data.track, flight.data.timestamp), sigma=sigma)

    ## IDENTIFY ALIGNED OVER RUNWAY SEGMENTS ##
    flight_segments = aligned_over_runway(flight, airport, rwy, scale)

    # If no aligned segments were found, return early to avoid errors downstream
    if not flight_segments:
        if debug: print("No aligned over-runway segments detected")
        return None, None, 0, None

    ## PROCESS ALIGNED OVER RUNWAY SEGMENTS ##
    results = process_flights_segments(flight, x1, x2, x3, x4, flight_segments, debug)

    ## PREPARE FLIGHT SEGMENTS ##
    fixed_length = 500
    
    if requires_phases:
        # Deep learning models (LSTM, CNN) need phases
        num_features_time = 4  # x1, x2, x3, x4
        phase_classes = ['CLIMB', 'DESCENT', 'LEVEL', 'NA']
        phase_to_idx = {p: i for i, p in enumerate(phase_classes)}
        phase_dim = len(phase_classes)
        num_features = num_features_time + phase_dim  # 4 + 4 = 8 total channels
    else:
        # Sklearn models (RF, LR) with engineered features - extract stats per channel
        num_features_time = 4
        phase_classes = ['CLIMB', 'DESCENT', 'LEVEL', 'NA']
        phase_to_idx = {p: i for i, p in enumerate(phase_classes)}
        phase_dim = len(phase_classes)
        # Will create 40 engineered features: (4 numeric + 4 phase one-hot) × 5 stats
    
    x_array = []
    results_with_phases = []  # Store results with resampled phases
    for i, (x1_seg, x2_seg, x3_seg, x4_seg, timestamps, flight_segment, index) in enumerate(results):
        
        segment_length = len(flight_segment.data)
        if debug: print(f"INITIAL SEGMENT LENGTH {i} : {segment_length}")
        
        # Extract phases for this segment
        phases_seg = flight_segment.data['phase'].values if 'phase' in flight_segment.data.columns else np.array(['NA'] * segment_length)

        if segment_length > fixed_length:
            # Downsample by slicing
            step = segment_length / fixed_length
            indices = (np.arange(fixed_length) * step).astype(int)
            x1_seg = x1_seg[indices]
            x2_seg = x2_seg[indices]
            x3_seg = x3_seg[indices]
            x4_seg = x4_seg[indices]
            phases_seg = phases_seg[indices]
            timestamps_resampled = np.array(timestamps)[indices]
        elif segment_length < fixed_length and segment_length > 1:
            # Upsample by interpolation
            # Sanitize timestamps: Convert to datetime, coerce errors to NaT, then drop NaT
            timestamps_dt = pd.to_datetime(timestamps, errors='coerce')  # Invalid values become NaT
            valid_mask = ~pd.isna(timestamps_dt)  # Mask for valid (non-NaT) timestamps
            
            if valid_mask.sum() == 0:
                raise ValueError("No valid timestamps found in the segment. Cannot resample.")

            # Filter all segment data to keep only valid timestamps
            x1_seg = x1_seg[valid_mask]
            x2_seg = x2_seg[valid_mask]
            x3_seg = x3_seg[valid_mask]
            x4_seg = x4_seg[valid_mask]
            phases_seg = phases_seg[valid_mask]
            timestamps_clean = timestamps_dt[valid_mask]

            # Update segment_length after filtering
            segment_length = len(x1_seg)

            if segment_length > 1:
                new_indices = np.linspace(0, segment_length - 1, fixed_length)
                x1_seg = np.interp(new_indices, np.arange(segment_length), x1_seg)
                x2_seg = np.interp(new_indices, np.arange(segment_length), x2_seg)
                x3_seg = np.interp(new_indices, np.arange(segment_length), x3_seg)
                x4_seg = np.interp(new_indices, np.arange(segment_length), x4_seg)
                # For phases (categorical), use nearest neighbor resampling
                phases_seg = phases_seg[np.round(new_indices).astype(int)]
                timestamps_numeric = timestamps_clean.astype('int64') / 1e9  # Convert to seconds since epoch (float seconds)
                timestamps_resampled = np.interp(new_indices, np.arange(segment_length), timestamps_numeric)
                # IMPORTANT: interpret numeric values explicitly as seconds since epoch to avoid 1970 artifacts
                timestamps_resampled = pd.to_datetime(timestamps_resampled, unit='s', utc=True)
            else:
                # After filtering, only one point left, repeat value
                x1_seg = np.full(fixed_length, x1_seg[0])
                x2_seg = np.full(fixed_length, x2_seg[0])
                x3_seg = np.full(fixed_length, x3_seg[0])
                x4_seg = np.full(fixed_length, x4_seg[0])
                phases_seg = np.full(fixed_length, phases_seg[0])
                timestamps_resampled = np.full(fixed_length, timestamps_clean.iloc[0])
        elif segment_length == 1:
            # Only one point, repeat value
            x1_seg = np.full(fixed_length, x1_seg[0])
            x2_seg = np.full(fixed_length, x2_seg[0])
            x3_seg = np.full(fixed_length, x3_seg[0])
            x4_seg = np.full(fixed_length, x4_seg[0])
            phases_seg = np.full(fixed_length, phases_seg[0])
            # timestamps is a pandas Series with arbitrary index labels; use positional access
            timestamps_resampled = np.full(fixed_length, timestamps.iloc[0])
        else:
            # Already correct length
            timestamps_resampled = np.array(timestamps)

        # Store resampled data with phases for plotting
        results_with_phases.append((x1_seg, x2_seg, x3_seg, x4_seg, pd.Series(timestamps_resampled), flight_segment, index, phases_seg))

        if requires_phases:
            # Deep learning models: build 3D array with shape (fixed_length, 8)
            segment_data = np.zeros((fixed_length, num_features))
            
            # Populate numeric features (channels 0-3)
            segment_data[:,0] = x1_seg
            segment_data[:,1] = x2_seg
            segment_data[:,2] = x3_seg
            segment_data[:,3] = x4_seg
            
            # Build one-hot phase channels (channels 4-7)
            one_hot = np.zeros((fixed_length, phase_dim), dtype=float)
            for t, lbl in enumerate(phases_seg):
                j = phase_to_idx.get(str(lbl), phase_to_idx['NA'])
                one_hot[t, j] = 1.0
            segment_data[:,4:4+phase_dim] = one_hot
            
            # Append processed segment data
            x_array.append(segment_data)
        else:
            # Engineered features for RF/LR: extract statistics per channel
            # Build full 8-channel array first (4 numeric + 4 phase one-hot)
            segment_data_full = np.zeros((fixed_length, 8))
            segment_data_full[:,0] = x1_seg
            segment_data_full[:,1] = x2_seg
            segment_data_full[:,2] = x3_seg
            segment_data_full[:,3] = x4_seg
            
            # Build one-hot phase channels
            one_hot = np.zeros((fixed_length, phase_dim), dtype=float)
            for t, lbl in enumerate(phases_seg):
                j = phase_to_idx.get(str(lbl), phase_to_idx['NA'])
                one_hot[t, j] = 1.0
            segment_data_full[:,4:8] = one_hot
            
            # Extract 5 statistics per channel: mean, std, min, max, median
            engineered_features = np.zeros(40)  # 8 channels × 5 stats
            for ch in range(8):
                channel_data = segment_data_full[:, ch]
                engineered_features[ch*5 + 0] = np.mean(channel_data)
                engineered_features[ch*5 + 1] = np.std(channel_data)
                engineered_features[ch*5 + 2] = np.min(channel_data)
                engineered_features[ch*5 + 3] = np.max(channel_data)
                engineered_features[ch*5 + 4] = np.median(channel_data)
            
            x_array.append(engineered_features)
    
    # Convert x_array to numpy array
    x_array = np.array(x_array)
    
    ## PREDICT USING THE LOADED MODEL ##
    if is_deep_learning:
        # Deep learning models (LSTM, CNN): expect 3D input (samples, timesteps, features)
        # Shape should be (num_samples, 500, 8)
        
        # Split into numeric and phase parts for scaling
        x_numeric = x_array[:, :, :4]  # First 4 channels
        x_phases = x_array[:, :, 4:]   # Last 4 channels (one-hot)
        
        # Scale ONLY numeric features
        nsamples, nt, nfeat = x_numeric.shape
        x_numeric_scaled = scaler.transform(x_numeric.reshape(-1, nfeat)).reshape(x_numeric.shape)
        
        # Recombine: [scaled numeric | unscaled one-hot phases]
        x_array_scaled = np.concatenate([x_numeric_scaled, x_phases], axis=2)
        
        # Predict
        y_pred_logits = model_obj.predict(x_array_scaled).flatten()
        # Apply sigmoid to convert logits to probabilities
        import tensorflow as tf
        y_pred = tf.nn.sigmoid(y_pred_logits).numpy()
    else:
        # Sklearn models (RF, LR): expect 2D engineered features (samples, 40)
        # Pipeline handles scaling internally, so pass raw engineered features
        if hasattr(model_obj, 'predict_proba'):
            y_pred = model_obj.predict_proba(x_array)[:, 1]
        else:
            y_pred = model_obj.predict(x_array)
            y_pred = np.array(y_pred).flatten()
    
    y_predbin = (y_pred > 0.5).astype(int)
    
    ## OPTIONAL OUTPUT : PLOT FLIGHT SEGMENTS ##
    if plot:
        # Use results_with_phases for plotting (includes resampled phase data)
        results_array = np.empty(len(results_with_phases), dtype=object)
        for idx, item in enumerate(results_with_phases):
            results_array[idx] = item
        plot_flight_segments(results_array, airport_obj, len(results_with_phases), y_pred)

    if debug: print(y_predbin)
    
    ## OUTPUT: NUMBER OF CIRCUITS ##
    ncircuits = len(y_predbin[y_predbin == 1])
    if ncircuits >=1:
        indexes = [(item[0], item[1]) for item, pred in zip(results[:, 6], y_predbin) if pred == 1]
        if debug: print(f"Number of airdrome circuits detected: {ncircuits}")
        return results, y_predbin, ncircuits, indexes
    else:
        if debug: print("No airdrome circuit detected")
        return results, y_predbin, 0, None

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
    airport_obj = airport if isinstance(airport, Airport) else airports[airport]
    rwy, rwy_scale = detect_runway(flight, airport_obj, scale, 10*scale)
    extreme1, extreme2, center = retrieve_runway_information(airport_obj, rwy)

    # DEFINE RADIUS #
    upper_coords = (47.562333, 8.986976) 
    radius = geodesic((center.latitude, center.longitude), upper_coords)

    if flight.aircraft is None:
        aircraft_type = get_aircraft_type(flight.data.callsign.iloc[0], method = 2, debug=debug)
    else:
        aircraft_type = get_aircraft_type(flight.aircraft.registration, method = 1, debug=debug)
    
    landing_lszt = flight.landing_at(airport_obj)      # landing bool
    takeoff_lszt = flight.takeoff_from(airport_obj)    # takeoff bool

    # DEPARTURE IDENTIFICATION (D) #
    if takeoff_lszt:
        climb_segment = flight.query(f"vertical_rate > {flight.data.vertical_rate.iloc[0]}")
        if climb_segment is not None and not climb_segment.data.empty:
            starting_flight = climb_segment.first('1 min')
    
            d_1 = starting_flight.distance(extreme1)
            t_1 = d_1.data.distance.idxmin()
            r_1 = d_1.data.loc[t_1]
            
            d_2 = starting_flight.distance(extreme2)
            t_2 = d_2.data.distance.idxmin()
            r_2 = d_2.data.loc[t_2]

            #d_24 = starting_flight.distance(extreme1).data.distance
            #d_06 = starting_flight.closest_point(extreme2)

            if r_1.timestamp > r_2.timestamp:
                rwy_takeoff = extreme1.name
                
                departure_time = r_1.timestamp
            else:
                rwy_takeoff = extreme2.name

                departure_time = r_2.timestamp

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
                print(f"TIME Threshold {extreme1.name} : {r_1.timestamp}")
                print(f"DISTANCE Threshold {extreme1.name}: {r_1.distance}")
                print(f"TIME Threshold {extreme2.name} : {r_2.timestamp}")
                print(f"DISTANCE Threshold {extreme2.name} : {r_2.distance}")
                print("------------------------------------------------")

    # VOLTE IDENTIFICATION (V) #
    _, _, ncircuits, indexes = find_aerodrome_circuits(flight, airport_obj, rwy,  model = 'cnn', scale=rwy_scale, debug=debug, plot=False) 
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
            d_1 = volte_flight.distance(extreme1)
            t_1 = d_1.data.distance.idxmin()
            r_1 = d_1.data.loc[t_1]
            
            d_2 = volte_flight.distance(extreme2)
            t_2 = d_2.data.distance.idxmin()
            r_2 = d_2.data.loc[t_2]

            if r_1.timestamp > r_2.timestamp:
                rwy_volte = extreme1.name
                route_volte = ''
                
                departure_time = r_1.timestamp
            else:
                rwy_volte = extreme2.name
                route_volte = ''               

            flight_data.append(
                ["LSZT", date[0], date[1], date[2], start_volte.strftime("%H%M"), "V", 
                4, aircraft_type, flight.data.callsign.iloc[0], rwy_volte, ncircuits*2, 0, 0, origin_dest, route_volte]
            )

    # ARRIVAL IDENTIFICATION (A) #
    if landing_lszt:
        last_change_index = flight.data.iloc[::-1].vertical_rate.ne(flight.data.vertical_rate.iloc[-1]).idxmax()
        ending_flight = flight.query(f"index <= {last_change_index}").last('1 min')

        d_1 = ending_flight.distance(extreme1)
        t_1 = d_1.data.distance.idxmin()
        r_1 = d_1.data.loc[t_1]
        
        d_2 = ending_flight.distance(extreme2)
        t_2 = d_2.data.distance.idxmin()
        r_2 = d_2.data.loc[t_2]

        if r_1.timestamp > r_2.timestamp:
            rwy_landing = extreme1.name
            
            landing_time = r_1.timestamp
        else:
            rwy_landing = extreme2.name

            landing_time = r_2.timestamp
    
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
            print(f"TIME Threshold {extreme1.name} : {r_1.timestamp}")
            print(f"DISTANCE Threshold {extreme1.name}: {r_1.distance}")
            print(f"TIME Threshold {extreme2.name} : {r_2.timestamp}")
            print(f"DISTANCE Threshold {extreme2.name} : {r_2.distance}")
            print("------------------------------------------------")
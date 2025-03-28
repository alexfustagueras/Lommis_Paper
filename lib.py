# Library of self-defined functions
import traffic
import joblib
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from traffic.core import Traffic, Flight
from traffic.core.mixins import PointMixin
from shapely.geometry import Point, Polygon
from tensorflow.keras.models import load_model

def generate_random_color():
    """
    Generate a random color in hex format
    """
    
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def inside_area(flight: Flight, box: list) -> Flight:
    """
    Filters the flight data to include only the points within the specified area.

    Parameters
    ----------
    flight : Flight
        Flight to be analyzed
    box : list 
        A box area with points A,B,C,D forming a rectangule in 2D plane, e.g. [A,B,C,D]

    Returns
    ----------
    Flight
        Cropped flight with only the datapoints inside the defined box
    """
    # NOTE: for some unknown reasong it's better to use a tuple of floats than shapely Geometry...
    west = min(box[0][1],box[3][1])
    south = min(box[0][0],box[1][0])
    east = max(box[1][1],box[2][1]) 
    north = max(box[2][0],box[3][0])

    if south > north: 
        oldsouth = south
        south = north
        north = oldsouth
    elif west > east:
        oldwest = west
        west = east
        east = oldwest
    
    return flight.inside_bbox([west,south,east,north]) # a tuple of floats (west, south, east, north)
 
def create_area(lat: float, long: float, r: float = 0.05) -> Polygon:
    """
    Given lattitude and longitude coordinates, it creates a Polygon area centered arround this coordinates with radius r

    Parameters
    ----------
    lat : float
        latitude coordinate
    long : float
        longitude coordinate
    r : float
        radius desired for the Polygon area
 
    Returns
    ----------
    Polygon
        Polygon area centered in (long,lat) point
    """
    point = Point(long, lat)
    buffer = point.buffer(r)
    polygon = Polygon(buffer.exterior.coords)
    return polygon

def normalize_track(track: list, time: list, window_size: int = 10) -> list:
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
    
    # Check for division by zero or constant data (no variance)
    if track_max == track_min:
        normalized_track = np.zeros_like(track_interpolated)
    else:
        normalized_track = (track_interpolated - track_min) / (track_max - track_min)
    
    return np.array(normalized_track)

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
            
    # at the end, add the final segment of the flight
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

def calculate_horizontal_angle(x1_lat, x1_long, x2_lat, x2_long, longitudes, latitudes):
    """
    Calculate the horizontal angle with respect to the midpoint of the runway's threshold using the angle between two vectors

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
    # Calculate the midpoint of the runway threshold
    mid_lat = (x1_lat + x2_lat) / 2
    mid_long = (x1_long + x2_long) / 2
    
    # Calculate vectors CP and AB
    delta_x_CP = longitudes - mid_long
    delta_y_CP = latitudes - mid_lat
    delta_x_AB = x2_long - x1_long
    delta_y_AB = x2_lat - x1_lat
    
    # Calculate the dot product and magnitudes of the vectors
    dot_product = delta_x_CP * delta_x_AB + delta_y_CP * delta_y_AB
    # cross_product = delta_x_CP * delta_y_AB - delta_y_CP * delta_x_AB  # Determines direction
    magnitude_CP = np.sqrt(delta_x_CP**2 + delta_y_CP**2)
    magnitude_AB = np.sqrt(delta_x_AB**2 + delta_y_AB**2)
    
    # Calculate the angle between the vectors in radians and convert to degrees
    cos_theta = dot_product / (magnitude_CP * magnitude_AB)
    angle = np.rad2deg(np.arccos(cos_theta))

    # angles = np.rad2deg(np.arctan2(cross_product, dot_product))
    
    return angle

def process_flights_segments(flight, x1, x2, x3, x4, segments, debug = False):
    """
    Process flight segments based on ILS segments and extract features
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
        
        start_index = flight.data.index.get_loc(flight_segment.data.index[0])
        end_index = flight.data.index.get_loc(flight_segment.data.index[-1]) + 1

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

def opposite_runway(rwy):
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

def create_runway_area(airport: str, rwy: str, scale: float = 1.0, debug: bool = False) -> list:
    """
    Creates a rectangle area around an specified runway of an airport.

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

    extreme1 = rwy
    extreme2 = opposite_runway(extreme1)

    df = airport.runways.data
    extreme1_info = df[df.name == extreme1]
    extreme2_info = df[df.name == extreme2]

    extreme1_lat = extreme1_info.latitude.iloc[0]
    extreme1_long = extreme1_info.longitude.iloc[0]
    extreme2_lat = extreme2_info.latitude.iloc[0]
    extreme2_long = extreme2_info.longitude.iloc[0]

    points = np.array([
        [extreme1_long - scale/1e3*np.sin(np.pi/2 - rwy_angle), extreme1_lat + scale/5e3*np.cos(np.pi/2 - rwy_angle)], # south west (A)
        [extreme2_long - scale/1e3*np.sin(np.pi/2 - rwy_angle), extreme2_lat + scale/5e3*np.cos(np.pi/2 - rwy_angle)], # south east (B)
        [extreme2_long + scale/1e3*np.sin(np.pi/2 - rwy_angle), extreme2_lat - scale/5e3*np.cos(np.pi/2 - rwy_angle)], # north east (C)
        [extreme1_long + scale/1e3*np.sin(np.pi/2 - rwy_angle), extreme1_lat - scale/5e3*np.cos(np.pi/2 - rwy_angle)]  # north west (D)
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
        airport of interest for the given flight
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
    segments = inside_area(flight, area) 
    if segments is not None:
        return detect_trajectories(segments, debug=debug)

def detect_runway(flight: Flight, airport: str, scale: float = 1.0):
    """
    Given a flight and an airport, detects on which runway has the flight been

    Parameters
    ----------
    flight : Flight
        Flight to be analyzed
    airport : str
        airport of interest for the given flight
    scale : float
        parameter that magnifies the area of capture around the runway
    Returns
    ----------
    str
        largest runway in which the flight has been, expressed as a runway number
    """
    exit = False
    largest_seg = 0

    available_rwys = airport.runways.data.name # list of available rwys
    for rwy in available_rwys:
        segments = aligned_over_runway(flight, airport, rwy, scale=scale) # find all segments in that rwy
        
        if segments is not None:
            # compute how many data you have in segments
            len_seg = 0
            for segment in segments:
                len_seg += len(segment.data)
            
            if len_seg > largest_seg:
                exit = True
                most_probable_rwy = rwy
                largest_seg = len_seg

    if exit == False:
        print("No valid runway detected.")
        return -1
    else:
        return most_probable_rwy
    
def retrieve_rwy_extremes(flight, airport, scale: float = 1.0):
    """
    Given a flight and an airport, it retrieves the runway extreme coordinates 
    from the most likely runway in which the flight has been

    Parameters
    ----------
    flight : Flight
        Flight to be analyzed
    airport : str
        airport of interest for the given flight
    Returns
    ----------
        extreme1.latitude
        extreme1.longitude
        extreme2.latitude
        extreme2.longitude
    """
    df = airport.runways.data

    extreme1 = detect_runway(flight, airport, scale)
    if extreme1 == -1:
        while extreme1 == -1 or scale <= 10:
            scale += 0.2
            extreme1 = detect_runway(flight, airport, scale)

    if extreme1 is None:
        return None, None, None, None
    
    extreme2 = opposite_runway(extreme1)

    extreme1_info = df[df.name == extreme1]
    extreme2_info = df[df.name == extreme2]

    return extreme1_info.latitude.iloc[0], extreme1_info.longitude.iloc[0], extreme2_info.latitude.iloc[0], extreme2_info.longitude.iloc[0]

def plot_flight_segments(results, airport, num_segments, predictions):
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

def find_airdrome_circuits(flight: Flight, airport: str = None, scale: float = 1.0, debug: bool = False, plot: bool = False):
    """
    Based on a trained Machine Learning model, it analyzes the flight and fins the number of detected airdrome circuits
    on the flight, as well as the time interval they ocurred

    Parameters
    ----------
    flight : Flight
        Flight to be analyzed
    airport : str = None
        ICAO code of the airport, if not provided it will just use the detected landing airport
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

    ## LOAD MACHINE LEARNING MODEL ##
    model = joblib.load("ML/Models/RandomForest/advanced_model_lommis_rf.pkl")

    extreme1 = PointMixin()
    extreme2 = PointMixin()
    extreme1.latitude, extreme1.longitude, extreme2.latitude, extreme2.longitude = retrieve_rwy_extremes(flight, airport, scale)
    if any(value is None for value in [extreme1.latitude, extreme1.longitude, extreme2.latitude, extreme2.longitude]):
        print("No airdrome circuit detected")
        return 0, None
    
    center = PointMixin()
    center.latitude = (extreme1.latitude + extreme2.latitude) / 2
    center.longitude = (extreme1.longitude + extreme2.longitude) / 2

    ## COMPUTE FEATURES ##
    x1 = np.array( flight.distance(center).data[["distance"]] ).flatten()
    x2 = np.array(calculate_horizontal_angle(extreme1.latitude, extreme1.longitude, extreme2.latitude, extreme2.longitude, 
                                            flight.data.longitude.values, flight.data.latitude.values) )
    x3 = np.array(flight.data.altitude)
    x4 = normalize_track(flight.data.track, flight.data.timestamp,0)
    
    ## IDENTIFY ALIGNED OVER RUNWAY SEGMENTS ##
    rwy = detect_runway(flight, airport, scale)
    if rwy == -1:
        while rwy == -1 or scale <= 10:
            scale += 0.2
            rwy = detect_runway(flight, airport, scale)

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
    
    # Convert x_array to a numpy array with the correct shape for the ML algorithm
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
        print("No airdrome circuit detected")
        return 0, None

# Work in progress...
def plot_flight_type(flight: Flight, airport: str, debug: bool = False):
    """
    Plots the types of flight phase detected versus the time of the flight, creates a "type" column on flight

    Parameters
    ----------
    flight : Flight
        Flight to be analyzed
    airport : str
        ICAO code of the airport
    d0 : float
        distance closer from the airport where start detecting circuits for valleys (in NM)
    d1 : float
        distance away from the airport where start detecting circuits for peaks (in NM)
    d2 : float
        distance away from the airport when stop detecting circuits (in NM)
    debug : bool
        debug toggle
    """
    flight.data['timestamp'] = flight.data['timestamp'].dt.tz_convert(None).astype('datetime64[ns]')
    flight.data['timestamp'] = flight.data['timestamp'].dt.tz_localize('UTC')

    flight = flight.phases()
    flight.data['type'] = 'N/A' # add the 'type' column with default value 'NA'
    
    # DEPARTURE (D : Departure (Abflug))
    phase_shift = (flight.data.phase.shift() == 'CLIMB') & (flight.data.phase != 'CLIMB')
    departure_index = phase_shift[1:].idxmax()

    flight.data.loc[:departure_index, 'type'] = 'D : Departure (Abflug)'  # set 'D' for departure phase

    # ARRIVAL (A : Arrival (Anflug))
    phase_shift = (flight.data.phase.shift() != flight.data.phase) & (flight.data.phase == 'DESCENT')
    descent_indices = phase_shift[phase_shift].index

    for index in np.flip(descent_indices):
        subsequent_phases = flight.data.loc[index+1:, 'phase']
        if not any(subsequent_phases.isin(['CLIMB'])):  # condition A: if there are not any climb after descent phase
            flight.data.loc[index:, 'type'] = 'A : Arrival (Anflug)'
            break  # stop when condition A is fulfilled

    # CIRCUIT VOLTE (V : Volte (Platzrunde))
    ncircuits, indexes = find_airdrome_circuits(flight, airport)

    if ncircuits >= 1:
        col_index = flight.data.columns.get_loc('type')
        for start_index, end_index in indexes:
            flight.data.iloc[start_index:end_index, col_index] = 'V : Volte (Platzrunde)'
            if debug:
                print(f"Identifying airdrome circuit between {flight.data.timestamp.iloc[start_index]} and {flight.data.timestamp.iloc[end_index]}")

    # plot
    color_scale = {'D : Departure (Abflug)': 'orange', 'A : Arrival (Anflug)': 'deepskyblue', 'V : Volte (Platzrunde)': 'teal', 'N/A': 'crimson'}

    fig, ax1 = plt.subplots(figsize=(20, 6))

    # Plot altitude
    ax1.plot(flight.data['timestamp'], flight.data['altitude'], label='Altitude', color='blue')
    ax1.set_ylabel('Altitude')
    ax1.set_xlabel('Timestamp')

    # Create a second y-axis for the type line
    ax2 = ax1.twinx()
    ax2.set_yticks([])  # Hide y-axis ticks for the second y-axis

    # Plot type line with different colors
    for t in ['N/A', 'A : Arrival (Anflug)', 'V : Volte (Platzrunde)', 'D : Departure (Abflug)']:
        subset = flight.data[flight.data['type'] == t]
        ax2.scatter(subset['timestamp'], [0] * len(subset), label=t, color=color_scale.get(t, 'black'))

    # Add vertical dashed lines and annotations
    for t in ['D : Departure (Abflug)', 'A : Arrival (Anflug)', 'V : Volte (Platzrunde)']:
        subset = flight.data[flight.data['type'] == t]
        if not subset.empty:
            ax1.axvline(x=subset['timestamp'].iloc[0], color=color_scale[t], linestyle='--')
            ax1.text(subset['timestamp'].iloc[0], max(flight.data['altitude']), t.split(':')[0], rotation=90, verticalalignment='bottom', color=color_scale[t])

    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    fig.legend(loc='upper right')
    plt.show()

    # fig, ax = plt.subplots(figsize=(20, 3))

    # for t in ['N/A', 'A : Arrival (Anflug)', 'V : Volte (Platzrunde)', 'D : Departure (Abflug)']:
    #     subset = flight.data[flight.data['type'] == t]
    #     ax.scatter(subset['timestamp'], subset['type'], label=t, color=color_scale.get(t, 'black'))

    # plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    # plt.legend()
    # plt.grid()
    # plt.show()
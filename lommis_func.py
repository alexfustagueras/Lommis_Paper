import csv
import lib
import traffic
from pyproj import Transformer
from cartes.crs import Lambert93
from geopy.distance import geodesic # type: ignore
from traffic.core.mixins import PointMixin
from traffic.data import airports, opensky, eurofirs
transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)

def determine_direction(entry_lat, entry_lon, center_lat, center_lon):
    """
    Determines the direction the aircraft is coming from based on entry coordinates 
    relative to the circle center.
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

def get_aircraft_type(registration_number, method = 1, debug=False):
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

def analyze_flight(flight, flight_data, date, debug = False ):

    # DEFINE RUNWAY EXTREME POINTS #
    extreme1 = PointMixin()
    extreme2 = PointMixin()
    extreme1.latitude = 47.5257
    extreme1.longitude = 9.0068
    extreme2.latitude = 47.5233
    extreme2.longitude = 8.9996
    extreme1.name = "Runway Threshold 24"
    extreme2.name = "Runway Threshold 06"

    # DEFINE CIRCLE CENTER #
    center = PointMixin()
    center.latitude = (extreme1.latitude + extreme2.latitude) / 2
    center.longitude = (extreme1.longitude + extreme2.longitude) / 2

    # DEFINE RADIUS #
    upper_coords = (47.562333, 8.986976) 
    radius = geodesic((center.latitude, center.longitude), upper_coords)

    if flight.aircraft is None:
        aircraft_type = get_aircraft_type(flight.data.callsign.iloc[0], method = 2, debug=debug)
    else:
        aircraft_type = get_aircraft_type(flight.aircraft.registration, method = 1, debug=debug)
    
    landing_lszt = flight.landing_at(airports["LSZT"])      # landing bool
    takeoff_lszt = flight.takeoff_from(airports["LSZT"])    # takeoff bool

    # DEPARTURE IDENTIFICATION (D) #
    if takeoff_lszt:
        starting_flight = flight.query(f"vertical_rate > {flight.data.vertical_rate.iloc[0]}").first('1 min')

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
    ncircuits, indexes = lib.find_airdrome_circuits(flight, airports["LSZT"], scale = 1, debug=debug, plot=False) 
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
                if takeoff_lszt: pos_landing = 1
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


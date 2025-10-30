import streamlit as st
# Set Streamlit page to wide layout
st.set_page_config(layout="wide")
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import glob
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from pathlib import Path
import io

# Imports from traffic library
from traffic.core import Traffic, Flight
from traffic.data import airports

from lommis_func import process_flights_segments, compute_alignment_angle, unwrap_track, aligned_over_runway, retrieve_runway_information, detect_runway
from scipy.ndimage import gaussian_filter1d

# Suppress warnings from the traffic library
logging.getLogger("traffic").setLevel(logging.ERROR)

# --- Utility Functions ---
@st.cache_data
def load_all_flights(base_path="Statistics/"):
    """
    Recursively finds and loads all flight data from a specified base path
    and returns a single Traffic object.
    """
    flight_objects = []
    # Match the user's working notebook patterns explicitly
    patterns = [
        os.path.join(base_path, '*.parquet'),
        os.path.join(base_path, 'Others', '*.parquet'),
        os.path.join(base_path, 'Others', 'PH_AXB', '*.parquet'),
    ]
    parquet_files = []
    for pat in patterns:
        parquet_files.extend(glob.glob(pat))
    
    if not parquet_files:
        st.error(f"No .parquet files found in '{base_path}'. Please check the folder structure.")
        return []

    for file_path in parquet_files:
        try:
            # Load a single flight from the parquet file
            flight = Flight.from_file(file_path)
            flight_objects.append(flight)
        except Exception as e:
            st.warning(f"Could not load flight from {file_path}: {e}")

    # Combine all individual flight objects into a single Traffic object
    if not flight_objects:
        return []
    
    traff_set = Traffic.from_flights(flight_objects)
    
    if 'timestamp' in traff_set.data.columns:
        traff_set.data['timestamp'] = traff_set.data['timestamp'].dt.tz_convert(None).astype('datetime64[ns]')
        traff_set.data['timestamp'] = traff_set.data['timestamp'].dt.tz_localize('UTC')

    return list(traff_set)

def load_flights_from_consolidated(parquet_path: str = 'Flights/flights_to_label.parquet'):
    """
    Load flights from a consolidated parquet built from Flights/to_label/*.pkl.
    Expects columns: flight_pickle (bytes). Optionally: airport_code, rwy, unique_flight_id.
    Returns a list of dicts with keys: flight, airport_code, rwy (when available).
    """
    try:
        import pickle as _pickle
        df = pd.read_parquet(parquet_path)
    except Exception:
        return []
        
    items = []
    for _, row in df.iterrows():
        try:
            flight = _pickle.loads(row['flight_pickle'])
        except Exception:
            continue
        inferred_landing = row.get('inferred_landing_airport', None) if 'inferred_landing_airport' in row.index else None
        inferred_takeoff = row.get('inferred_takeoff_airport', None) if 'inferred_takeoff_airport' in row.index else None
        unique_flight_id = row.get('unique_flight_id', None) if 'unique_flight_id' in row.index else None
        entry = {
            'flight': flight,
            'inferred_landing_airport': inferred_landing,
            'inferred_takeoff_airport': inferred_takeoff,
            'unique_flight_id': unique_flight_id,
        }
        items.append(entry)
    return items

@st.cache_data
def process_data_for_labeling(_flight_list):
    """
    Processes all flights according to the user's provided workflow.
    This function processes the data once and caches the results for efficiency.
    """
    all_segments_to_label = []
    fixed_length = 500
    
    total = len(_flight_list)
    st.markdown(f" ")
    progress_bar = st.progress(0, text=f"Processing 0/{total} flights...")
    
    for idx, item in enumerate(_flight_list):
        try:
            # Support entries that carry precomputed airport/runway and the flight object
            if isinstance(item, dict) and 'flight' in item:
                flight = item['flight']
                inferred_landing = item.get('inferred_landing_airport')
                inferred_takeoff = item.get('inferred_takeoff_airport')
                unique_flight_id = item.get('unique_flight_id')
            else:
                flight = item
                inferred_landing = None
                inferred_takeoff = None
                unique_flight_id = None

            # Basic sanity checks
            if flight is None:
                print(f"Skipping flight at index {idx}: flight object is None")
                continue
            if not hasattr(flight, 'data') or flight.data is None or flight.data.empty:
                print(f"Skipping flight {unique_flight_id or idx}: empty or missing data")
                continue
            
            candidates = []
            
            # Try landing airport
            if inferred_landing:
                try:
                    apt_landing = airports[inferred_landing]
                    rwy_landing, scale_landing = detect_runway(flight, apt_landing)
                    if rwy_landing != -1:
                        # Count aligned segments at this airport
                        segments_landing = aligned_over_runway(flight, apt_landing, rwy_landing, scale=scale_landing)
                        n_segments_landing = len(segments_landing) if segments_landing else 0
                        candidates.append({
                            'airport': apt_landing,
                            'airport_code': inferred_landing,
                            'rwy': rwy_landing,
                            'n_segments': n_segments_landing,
                            'scale': scale_landing
                        })
                except Exception as e:
                    print(f"Error processing landing airport {inferred_landing} for flight {unique_flight_id or idx}: {e}")
            
            # Try takeoff airport
            if inferred_takeoff and inferred_takeoff != inferred_landing:
                try:
                    apt_takeoff = airports[inferred_takeoff]
                    rwy_takeoff, scale_takeoff = detect_runway(flight, apt_takeoff)
                    if rwy_takeoff != -1:
                        # Count aligned segments at this airport
                        segments_takeoff = aligned_over_runway(flight, apt_takeoff, rwy_takeoff, scale=scale_takeoff)
                        n_segments_takeoff = len(segments_takeoff) if segments_takeoff else 0
                        candidates.append({
                            'airport': apt_takeoff,
                            'airport_code': inferred_takeoff,
                            'rwy': rwy_takeoff,
                            'n_segments': n_segments_takeoff,
                            'scale': scale_takeoff
                        })
                except Exception as e:
                    print(f"Error processing takeoff airport {inferred_takeoff} for flight {unique_flight_id or idx}: {e}")
            
            # Select the airport with more aligned segments
            if not candidates:
                print(f"Skipping flight {unique_flight_id or idx}: no valid airport/runway detected")
                continue
            
            # Sort by number of segments (descending) and pick the best
            best = max(candidates, key=lambda c: c['n_segments'])
            
            # Skip if no segments found at any airport
            if best['n_segments'] == 0:
                print(f"Skipping flight {unique_flight_id or idx}: no aligned segments at any detected airport")
                continue
            
            # Use the best airport/runway
            airport = best['airport']
            airport_code_used = best['airport_code']
            rwy = best['rwy']
            scale_used = best['scale']

            extreme1, extreme2, center = retrieve_runway_information(airport, rwy)

            # Ensure phases column is available on the flight
            try:
                flight = flight.phases()
            except Exception:
                pass

            # --- 1. COMPUTE FEATURES ---
            sigma = 1.0
            x1 = gaussian_filter1d(np.array( flight.distance(center).data[["distance"]] ).flatten(), sigma=sigma)
            x2 = gaussian_filter1d(np.array(compute_alignment_angle(extreme1.latitude, extreme1.longitude, extreme2.latitude, extreme2.longitude, 
                                                    flight.data.longitude.values, flight.data.latitude.values)), sigma=sigma)
            x3 = gaussian_filter1d(np.array(flight.data.altitude), sigma=sigma)
            x4 = gaussian_filter1d(unwrap_track(flight.data.track, flight.data.timestamp), sigma=sigma)

            # If normalization returned None (no valid track points), skip flight
            if x4 is None:
                print(f"Skipping flight {unique_flight_id or idx}: unwrapped track is None")
                continue
            
            # --- 2. IDENTIFY ALIGNED OVER RUNWAY SEGMENTS ---
            if airport_code_used in ["LIPQ", "EGPK", "EBLG"]:
                scale = max(scale_used, 1.2)
            elif flight.callsign in ["HBETG"]:
                scale = 2.0
            else:
                scale = max(scale_used, 1.1)
            
            flight_segments = aligned_over_runway(flight, airport, rwy, scale)

            # If no aligned segments were found, return early to avoid errors downstream
            if not flight_segments:
                print("No aligned over-runway segments detected")
                continue

            # # --- 3. PROCESS ALIGNED OVER RUNWAY SEGMENTS ---
            results = process_flights_segments(flight, x1, x2, x3, x4, flight_segments, debug=False)

            # # --- 4. PREPARE FLIGHT SEGMENTS (RESAMPLING) ---
            ikeep = 0
            for (x1_seg, x2_seg, x3_seg, x4_seg, timestamps, flight_segment, index) in results:
                segment_length = len(x1_seg)

                # Extract phases for this segment (sequence preserved)
                if flight_segment is not None and hasattr(flight_segment, 'data') and 'phase' in flight_segment.data.columns:
                    phases_seg = flight_segment.data['phase'].values
                else:
                    phases_seg = np.array(['NA'] * segment_length)

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
                    new_indices = np.linspace(0, segment_length - 1, fixed_length)
                    x1_seg = np.interp(new_indices, np.arange(segment_length), x1_seg)
                    x2_seg = np.interp(new_indices, np.arange(segment_length), x2_seg)
                    x3_seg = np.interp(new_indices, np.arange(segment_length), x3_seg)
                    x4_seg = np.interp(new_indices, np.arange(segment_length), x4_seg)
                    # For phases (categorical), use nearest neighbor resampling
                    phases_seg = phases_seg[np.round(new_indices).astype(int)]
                    # Clean timestamps and convert to seconds since epoch for stable interpolation
                    timestamps_dt = pd.to_datetime(timestamps, errors='coerce')
                    valid_mask = ~pd.isna(timestamps_dt)
                    if valid_mask.sum() == 0:
                        # Fallback: generate a simple range if timestamps are unusable
                        timestamps_dt = pd.to_datetime(np.arange(segment_length), unit='s', utc=True)
                        valid_mask = np.ones(segment_length, dtype=bool)
                    timestamps_clean = timestamps_dt[valid_mask]
                    t_numeric = (timestamps_clean.astype('int64') / 1e9).to_numpy()
                    # Align new_indices to valid range
                    new_indices_valid = np.linspace(0, len(timestamps_clean) - 1, fixed_length)
                    t_resampled = np.interp(new_indices_valid, np.arange(len(timestamps_clean)), t_numeric)
                    timestamps_resampled = pd.to_datetime(t_resampled, unit='s', utc=True)
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

                # Extract original indices from the segment (index contains [start_index-1, end_index])
                segment_start_idx = index[0] + 1 if isinstance(index, (list, tuple)) and len(index) >= 2 else None
                segment_end_idx = index[1] if isinstance(index, (list, tuple)) and len(index) >= 2 else None

                all_segments_to_label.append({
                    'segment_data': (x1_seg, x2_seg, x3_seg, x4_seg, pd.Series(pd.to_datetime(timestamps_resampled).round('ms')), flight_segment, index, phases_seg),
                    'unique_flight_id': unique_flight_id,
                    'segment_num': ikeep,
                    'flight_pickle': pickle.dumps(flight),
                    'airport_code': airport_code_used,
                    'rwy': rwy,
                    'segment_start_idx': segment_start_idx,
                    'segment_end_idx': segment_end_idx
                })

                ikeep = ikeep + 1
            
        except Exception as e:
            unique_flight_id_str = unique_flight_id if unique_flight_id is not None else f"idx_{idx}"
            print(f"Error processing flight {unique_flight_id_str}: {e.__class__.__name__} - {e}")

        # Update progress bar inside the loop. This is the key.
        progress_bar.progress((idx + 1) / total, text=f"Processing {idx + 1}/{total} flights...")
            
    # Remove the progress bar once done
    progress_bar.empty()
    return all_segments_to_label

def plot_single_segment(result, airport, title="Flight Segment"):
    """
    Plots a single comprehensive visualization of a flight segment using Plotly.
    """
    fig = make_subplots(
        rows=1, 
        cols=5,
        subplot_titles=[
            'Distance [NM]', 
            'RLAA [deg.]',
            'Local Altitude [ft.]', 
            'Unwrapped Track [deg.]', 
            'Flight Path'
        ],
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "geo"}]]
    )

    # Normalize timestamps to ms to avoid nanosecond warnings and ensure consistent dtype
    timestamps = pd.to_datetime(result[4], errors='coerce')
    try:
        timestamps = timestamps.dt.round('ms')
    except Exception:
        pass

    # Optional phases sequence if present
    phases = result[7] if len(result) > 7 else None

    # Time series plots (handle plain numpy arrays / lists)
    try:
        # Add phase background rectangles per contiguous run (like lommis_func)
        phase_colors = {
            'CLIMB': 'rgba(192,255,192,0.5)',
            'DESCENT': 'rgba(255,214,214,0.5)',
            'LEVEL': 'rgba(255,240,122,0.5)',
            'NA': 'rgba(224,224,224,0.5)',
        }

        def add_phase_vrects(col_idx: int):
            if phases is None or len(phases) == 0:
                return
            ts = timestamps
            start = 0
            current = phases[0]
            for k in range(1, len(phases) + 1):
                if k == len(phases) or phases[k] != current:
                    x0 = ts.iloc[start] if hasattr(ts, 'iloc') else ts[start]
                    x1 = ts.iloc[k - 1] if hasattr(ts, 'iloc') else ts[k - 1]
                    color = phase_colors.get(str(current), 'rgba(224,224,224,0.5)')
                    fig.add_vrect(
                        x0=x0, x1=x1,
                        fillcolor=color, opacity=0.2, line_width=0, layer='below',
                        row=1, col=col_idx, exclude_empty_subplots=False,
                    )
                    if k < len(phases):
                        start = k
                        current = phases[k]

        add_phase_vrects(1)
        add_phase_vrects(2)
        add_phase_vrects(3)
        add_phase_vrects(4)

        fig.add_trace(go.Scatter(x=timestamps, y=result[0], mode='lines', name='Distance', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=timestamps, y=result[1], mode='lines', name='RLAA'), row=1, col=2)
        fig.add_trace(go.Scatter(x=timestamps, y=result[2], mode='lines', name='Geo Altitude', showlegend=False), row=1, col=3)
        fig.add_trace(go.Scatter(x=timestamps, y=result[3], mode='lines', name='Unwrapped Track', showlegend=False), row=1, col=4)
    except Exception:
        # fallback: plot against sample index
        idx = list(range(len(result[0])))
        fig.add_trace(go.Scatter(x=idx, y=result[0], mode='lines', name='Distance'), row=1, col=1)
        fig.add_trace(go.Scatter(x=idx, y=result[1], mode='lines', name='RLAA'), row=1, col=2)
        fig.add_trace(go.Scatter(x=idx, y=result[2], mode='lines', name='Geo Altitude'), row=1, col=3)
        fig.add_trace(go.Scatter(x=idx, y=result[3], mode='lines', name='Unwrapped Track'), row=1, col=4)

    # Geographic flight path (only if flight_segment provided)
    flight_segment = result[5] if len(result) > 5 else None
    if flight_segment is not None:
        try:
            fig.add_trace(go.Scattergeo(
                lon=flight_segment.data.longitude,
                lat=flight_segment.data.latitude,
                mode='lines',
                line=dict(width=2, color='red'),
                name='Flight Path', showlegend=False,
            ), row=1, col=5)
        except Exception:
            # if flight_segment doesn't have geo data, skip map
            pass
    
    # Add runway locations if airport data is available
    if airport and hasattr(airport, 'runways'):
        df = airport.runways.data
        # Plot each runway as a line between its two ends
        for idx in range(0, len(df), 2):
            if idx + 1 < len(df):
                fig.add_trace(go.Scattergeo(
                    lon=[df.longitude.iloc[idx], df.longitude.iloc[idx+1]],
                    lat=[df.latitude.iloc[idx], df.latitude.iloc[idx+1]],
                    mode='lines',
                    line=dict(width=3, color='black'),
                    name=f'Runway {df.name.iloc[idx]}' if 'name' in df.columns else 'Runway',
                    opacity=0.75, showlegend=False,
                ), row=1, col=5)

    # Update layout for the map plot with more zoom
    if flight_segment:
        fig.update_geos(
            projection_type="equirectangular",
            landcolor='rgb(243, 243, 243)',
            lonaxis_range=[min(flight_segment.data.longitude)-0.02, max(flight_segment.data.longitude)+0.02],
            lataxis_range=[min(flight_segment.data.latitude)-0.02, max(flight_segment.data.latitude)+0.02],
            row=1,
            col=5
        )

    # Add legend entries for phases (legend-only markers)
    phase_colors = {
        'CLIMB': 'rgba(192,255,192,0.5)',
        'DESCENT': 'rgba(255,214,214,0.5)',
        'LEVEL': 'rgba(255,240,122,0.5)',
        'NA': 'rgba(224,224,224,0.5)',
    }
    for _name, _color in phase_colors.items():
        fig.add_trace(go.Scatter(x=[], y=[], mode='markers', marker=dict(size=10, color=_color, symbol='square'), name=_name, hoverinfo='skip'))

    # Final layout configuration
    fig.update_layout(
        title=f'Flight Segment Analysis: {title}',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        height=400,
        width=1800,
    )

    st.plotly_chart(fig, use_container_width=True)

def get_labeled_parquet_bytes(labeled_list):
    """
    Convert the in-memory labeled list to parquet bytes for download.
    Always returns a bytes object (empty dataframe if list is empty).
    """
    try:
        # Ensure pandas is available in this scope
        if not labeled_list:
            df = pd.DataFrame(columns=["flight_id", "label", "features"])
        else:
            # Make a DataFrame; ensure consistent column ordering
            df = pd.DataFrame(labeled_list)

        buffer = io.BytesIO()
        # Use parquet format
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        logging.exception(e)
        # Fallback: return empty parquet bytes
        buffer = io.BytesIO()
        pd.DataFrame(columns=["flight_id", "label", "features"]).to_parquet(buffer, index=False)
        buffer.seek(0)
        return buffer.read()

# --- Main Streamlit Application ---
def main():
    st.title("🛩️ Flight Segment Labeling App")
    st.markdown("Use this app to manually label flight segments as 'Traffic Circuit' or 'Not a Circuit'.")
    st.markdown("---")
    # Sidebar: choose consolidated parquet path (for quick samples/tests)
    with st.sidebar:
        st.markdown("### Data source")
        cons_default = st.session_state.get('consolidated_parquet_path', 'Flights/flights_to_label.parquet')
        cons_path = st.text_input("Consolidated parquet path", value=cons_default, help="Path to Flights/flights_to_label.parquet or a sample like Flights/flights_to_label_5.parquet")
        st.session_state.consolidated_parquet_path = cons_path
    # --- Load existing labeled parquet (optional) ---
    with st.expander("Load existing labeled parquet", expanded=False):
        st.markdown("### Load existing labeled parquet")
        labeled_path = st.text_input("Path to labeled parquet (leave blank for default)", value="labeled_ground_truth.parquet")
        filter_mode = st.selectbox("Review mode", options=["Continue labeling (merge with full list)", "All", "Only labeled", "Only traffic_circuit", "Only not_a_circuit"], index=0)
        if st.button("Load labeled file"):
            p = Path(labeled_path)
            if not p.exists():
                st.error(f"File not found: {labeled_path}")
            else:
                try:
                    df = pd.read_parquet(str(p))
                    # Build segment list and labels from expected 'features' nested dict format
                    full_segments = []
                    full_labels = []
                    for idx, row in df.iterrows():
                        feats = row.get('features') if 'features' in row.index else row.get('feature') if 'feature' in row.index else None
                        
                        # Initialize with empty arrays as fallback
                        x1 = np.array([])
                        x2 = np.array([])
                        x3 = np.array([])
                        x4 = np.array([])
                        phases_loaded = None
                        
                        if isinstance(feats, dict):
                            # Safely get values, ensuring they're iterable
                            x1_raw = feats.get('x1', [])
                            x2_raw = feats.get('x2', [])
                            x3_raw = feats.get('x3', [])
                            x4_raw = feats.get('x4', [])
                            
                            x1 = np.array(x1_raw) if x1_raw is not None and hasattr(x1_raw, '__iter__') else np.array([])
                            x2 = np.array(x2_raw) if x2_raw is not None and hasattr(x2_raw, '__iter__') else np.array([])
                            x3 = np.array(x3_raw) if x3_raw is not None and hasattr(x3_raw, '__iter__') else np.array([])
                            x4 = np.array(x4_raw) if x4_raw is not None and hasattr(x4_raw, '__iter__') else np.array([])
                            
                            phase_raw = feats.get('phase', [])
                            if phase_raw is not None and hasattr(phase_raw, '__iter__') and not isinstance(phase_raw, str):
                                phases_loaded = np.array(phase_raw)
                        else:
                            # Try older column layout - safely access values
                            dist_raw = row.get('distance')
                            hangle_raw = row.get('hangle')
                            alt_raw = row.get('altitude')
                            track_raw = row.get('track')
                            
                            x1 = np.array(dist_raw) if dist_raw is not None and hasattr(dist_raw, '__iter__') else np.array([])
                            x2 = np.array(hangle_raw) if hangle_raw is not None and hasattr(hangle_raw, '__iter__') else np.array([])
                            x3 = np.array(alt_raw) if alt_raw is not None and hasattr(alt_raw, '__iter__') else np.array([])
                            x4 = np.array(track_raw) if track_raw is not None and hasattr(track_raw, '__iter__') else np.array([])

                        # Timestamps may not be present; use sample indices as fallback
                        length = max(len(x1), len(x2), len(x3), len(x4), 1)
                        timestamps = list(range(length))

                        if phases_loaded is not None and len(phases_loaded) == length:
                            seg_tuple = (x1, x2, x3, x4, timestamps, None, 0, phases_loaded)
                        else:
                            seg_tuple = (x1, x2, x3, x4, timestamps, None, 0)
                        full_segments.append({'segment_data': seg_tuple, 'flight_info': row.get('flight_id', f'loaded_{idx}'), 'airport': airports['LSZT']})
                        # Capture identifiers for robust matching later
                        full_labels.append({
                            'flight_id': row.get('flight_id', None) or row.get('unique_flight_id', f'loaded_{idx}'),
                            'unique_flight_id': row.get('unique_flight_id', None),
                            'segment_num': row.get('segment_num', None),
                            'label': row.get('label'),
                            'features': feats
                        })

                    # Apply filter
                    if filter_mode == 'Continue labeling (merge with full list)':
                        # Merge loaded labels into the full segment list from session state
                        if 'segments' in st.session_state and st.session_state.segments:
                            # Create a mapping of (unique_flight_id, segment_num) -> label row
                            label_map = {}
                            for i, lab in enumerate(full_labels):
                                uid = lab.get('unique_flight_id') or lab.get('flight_id')
                                segn = lab.get('segment_num')
                                lab_label = lab.get('label')
                                try:
                                    segn_int = int(segn) if segn is not None and not (isinstance(segn, float) and np.isnan(segn)) else None
                                except Exception:
                                    segn_int = None
                                # Only map entries that actually have a label (traffic_circuit / not_a_circuit / omitted)
                                if uid is not None and (lab_label is not None) and (not pd.isna(lab_label)):
                                    label_map[(uid, segn_int)] = lab
                            
                            # Merge labels into labeled_data_list
                            merged_labels = []
                            for seg_info in st.session_state.segments:
                                unique_id = seg_info.get('unique_flight_id')
                                seg_num = seg_info.get('segment_num', None)
                                try:
                                    seg_num_int = int(seg_num) if seg_num is not None else None
                                except Exception:
                                    seg_num_int = None

                                # Check if this segment was already labeled (match by flight + segment)
                                lab_data = label_map.get((unique_id, seg_num_int)) or label_map.get((unique_id, None))

                                if lab_data is not None and lab_data.get('label') is not None and not pd.isna(lab_data.get('label')):
                                    # Found a match, add to merged labels with existing label
                                    merged_labels.append({
                                        'unique_flight_id': unique_id,
                                        'segment_num': seg_num_int,
                                        'flight_pickle': seg_info.get('flight_pickle'),
                                        'airport_code': seg_info.get('airport_code'),
                                        'rwy': seg_info.get('rwy'),
                                        'segment_start_idx': seg_info.get('segment_start_idx'),
                                        'segment_end_idx': seg_info.get('segment_end_idx'),
                                        'label': lab_data.get('label'),
                                        'features': lab_data.get('features')
                                    })
                                else:
                                    # Not labeled yet, add empty placeholder
                                    merged_labels.append({
                                        'unique_flight_id': unique_id,
                                        'segment_num': seg_num_int,
                                        'flight_pickle': seg_info.get('flight_pickle'),
                                        'airport_code': seg_info.get('airport_code'),
                                        'rwy': seg_info.get('rwy'),
                                        'segment_start_idx': seg_info.get('segment_start_idx'),
                                        'segment_end_idx': seg_info.get('segment_end_idx'),
                                        'label': None,
                                        'features': None
                                    })
                            
                            st.session_state.labeled_data_list = merged_labels
                            
                            # Find first unlabeled segment to jump to
                            first_unlabeled = 0
                            for i, lab in enumerate(merged_labels):
                                if lab.get('label') is None or pd.isna(lab.get('label')):
                                    first_unlabeled = i
                                    break
                            
                            st.session_state.current_index = first_unlabeled
                            st.session_state.loaded_original_df = df
                            num_already_labeled = sum(1 for l in merged_labels if l.get('label') and pd.notna(l.get('label')))
                            st.success(f"✅ Merged {num_already_labeled} labeled segments into full list of {len(st.session_state.segments)}. Jumping to first unlabeled segment #{first_unlabeled + 1}")
                            st.rerun()
                        else:
                            st.error("No full segment list found. Please close this expander and let the app load all segments first, then reload this labeled file.")
                            return
                    
                    elif filter_mode == 'All':
                        segs = full_segments
                        labels = full_labels
                        original_indices = list(range(len(full_segments)))
                    else:
                        segs = []
                        labels = []
                        original_indices = []
                        for i, lab in enumerate(full_labels):
                            lab_val = lab.get('label')
                            if filter_mode == 'Only labeled' and pd.notna(lab_val):
                                segs.append(full_segments[i])
                                labels.append(lab)
                                original_indices.append(i)
                            elif filter_mode == 'Only traffic_circuit' and lab_val == 'traffic_circuit':
                                segs.append(full_segments[i])
                                labels.append(lab)
                                original_indices.append(i)
                            elif filter_mode == 'Only not_a_circuit' and lab_val == 'not_a_circuit':
                                segs.append(full_segments[i])
                                labels.append(lab)
                                original_indices.append(i)

                    st.session_state.segments = segs
                    st.session_state.labeled_data_list = labels
                    st.session_state.current_index = 0
                    st.session_state.loaded_original_df = df
                    st.session_state.loaded_original_indices = original_indices
                    st.success(f"Loaded {len(segs)} segments (filtered from {len(full_segments)} rows) from {p.name}")
                except Exception as e:
                    logging.exception(e)
                    st.error(f"Failed to load parquet: {e}")
    
    # --- State Initialization ---
    if 'segments' not in st.session_state:
        with st.spinner("Loading and processing flights (consolidated if available)..."):
            # Prefer consolidated parquet (fast, deterministic). Fallback to Statistics loader.
            all_flights = load_flights_from_consolidated(st.session_state.get('consolidated_parquet_path', 'Flights/flights_to_label.parquet'))
            if not all_flights:
                all_flights = load_all_flights()
                st.info(f"Loaded {len(all_flights)} flights from Statistics folder.")
            else:
                st.info(f"Loaded {len(all_flights)} flights from {st.session_state.consolidated_parquet_path}")

            st.session_state.segments = process_data_for_labeling(all_flights)
            st.session_state.current_index = 0
            st.session_state.labeled_data_list = []
            st.session_state.loaded_original_df = None
            st.session_state.loaded_original_indices = None

    segments_to_label = st.session_state.segments
    current_index = st.session_state.current_index

    # --- Labeled counts and percentages ---
    total_segments = len(segments_to_label)
    labeled_list = st.session_state.labeled_data_list
    
    # Count only entries with actual labels (not None/NaN)
    labeled_count = sum(1 for d in labeled_list if d.get('label') and pd.notna(d.get('label')))
    traffic_count = sum(1 for d in labeled_list if d.get('label') == 'traffic_circuit')
    not_count = sum(1 for d in labeled_list if d.get('label') == 'not_a_circuit')
    pct_labeled = (labeled_count / total_segments * 100) if total_segments else 0.0
    traffic_pct = (traffic_count / labeled_count * 100) if labeled_count else 0.0
    not_pct = (not_count / labeled_count * 100) if labeled_count else 0.0

    # Show metrics in a compact row
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Total segments", total_segments)
    mcol2.metric("Labeled", f"{labeled_count} ({pct_labeled:.1f}%)")
    mcol3.metric("Traffic circuits", f"{traffic_count} ({traffic_pct:.1f}%)")
    mcol4.metric("Not circuits", f"{not_count} ({not_pct:.1f}%)")


    # Reliable save buttons
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save ONLY labeled rows → EMERGENCY_SAVE.parquet"):
            df_all = pd.DataFrame(st.session_state.get('labeled_data_list', []))
            if not df_all.empty and 'label' in df_all.columns:
                df_labeled = df_all[df_all['label'].notna()].copy()
            else:
                df_labeled = df_all
            df_labeled.to_parquet("EMERGENCY_SAVE.parquet", index=False)
            st.success(f"Saved {len(df_labeled)} labeled rows to EMERGENCY_SAVE.parquet")
    with c2:
        if st.button("🧾 Save FULL review state → EMERGENCY_SAVE_FULL.parquet"):
            df_full = pd.DataFrame(st.session_state.get('labeled_data_list', []))
            df_full.to_parquet("EMERGENCY_SAVE_FULL.parquet", index=False)
            st.success(f"Saved full state with {len(df_full)} rows to EMERGENCY_SAVE_FULL.parquet")

    # --- Jump / edit controls ---
    with st.expander("Jump to / Edit a segment", expanded=False):
        max_idx = max(0, total_segments - 1)
        # ensure default value is within allowed range (Streamlit raises if value > max_value)
        safe_default = min(current_index, max_idx)
        jump_idx = st.number_input("Go to segment index (0-based)", min_value=0, max_value=max_idx, value=safe_default, step=1)
        replace_mode = st.checkbox("Replace existing label when labeling this segment", value=False)
        if st.button("Go"):
            st.session_state.current_index = int(jump_idx)
            st.session_state.replace_mode = bool(replace_mode)
            st.rerun()

    # Ensure replace_mode exists in session_state
    if 'replace_mode' not in st.session_state:
        st.session_state.replace_mode = False

    if current_index >= len(segments_to_label):
        st.success("🎉 All segments have been labeled!")
        if st.session_state.labeled_data_list:
            labeled_df = pd.DataFrame(st.session_state.labeled_data_list)
            st.download_button(
                label="Download Labeled Data",
                data=labeled_df.to_parquet(index=False),
                file_name="labeled_ground_truth.parquet",
                mime="application/octet-stream",
                key="download_success"
            )
        return

    # --- Labeling UI ---
    current_segment_info = segments_to_label[current_index]
    current_segment_data = current_segment_info['segment_data']
    airport_code = current_segment_info.get('airport_code')
    airport_data = airports[airport_code] if airport_code else None
    
    st.write(f"**Segment {current_index + 1}/{len(segments_to_label)}**")

    # Extract callsign and timestamp from the flight segment (robust when flight_segment is None)
    flight_segment = current_segment_data[5] if len(current_segment_data) > 5 else None
    if flight_segment is not None:
        callsign = getattr(flight_segment, 'callsign', None)
        try:
            timestamp = flight_segment.data.timestamp.iloc[0] if hasattr(flight_segment, 'data') and 'timestamp' in flight_segment.data else None
        except Exception:
            timestamp = None
    else:
        callsign = None
        # Use first resampled timestamp as representative when flight_segment missing
        try:
            ts_candidate = current_segment_data[4]
            timestamp = pd.to_datetime(ts_candidate[0]) if len(ts_candidate) else None
        except Exception:
            timestamp = None

    info_str = f"Callsign: {callsign if callsign else 'N/A'} | Segment: {current_index + 1} | Time: {timestamp if timestamp is not None else 'N/A'}"
    st.write(f"Flight Info: {info_str}")
    # Compose a safe name
    ts_name = f"{timestamp.hour}:{timestamp.minute}" if timestamp is not None else 'N/A'
    cs_name = callsign if callsign else 'N/A'
    st.write(f"Name : flight_segment_{cs_name}_{current_index + 1}_{ts_name}")

    # Show matching keys to verify against labeled parquet
    st.caption("Key to match in labeled parquet")
    st.code(
        {
            'unique_flight_id': current_segment_info.get('unique_flight_id', None),
            'segment_num': current_segment_info.get('segment_num', None)
        },
        language='json'
    )

    plot_single_segment(current_segment_data, airport_data, title=f"Segment {current_index + 1}")

    st.markdown("### Is this a traffic circuit?")

    # Show current segment label status and option to skip labeled when advancing
    current_label = None
    if current_index < len(st.session_state.labeled_data_list):
        current_label = st.session_state.labeled_data_list[current_index].get('label')
    status_text = f"Status: labeled as '{current_label}'" if (current_label is not None and not pd.isna(current_label)) else "Status: unlabeled"
    st.caption(status_text)
    if 'skip_labeled' not in st.session_state:
        st.session_state.skip_labeled = True
    st.session_state.skip_labeled = st.checkbox("Skip labeled segments when advancing", value=st.session_state.skip_labeled)

    col1, col2, col3, col4 = st.columns(4)
    
    # Ensure helper lists exist
    if 'omitted_entries' not in st.session_state:
        st.session_state.omitted_entries = []

    with col1:
        if st.button("✅ Yes, Traffic Circuit", use_container_width=True):
            seg_info = current_segment_info
            entry = {
                'unique_flight_id': seg_info.get('unique_flight_id'),
                'segment_num': seg_info.get('segment_num', current_index),
                'flight_pickle': seg_info.get('flight_pickle'),
                'airport_code': seg_info.get('airport_code'),
                'rwy': seg_info.get('rwy'),
                'segment_start_idx': seg_info.get('segment_start_idx'),
                'segment_end_idx': seg_info.get('segment_end_idx'),
                'label': 'traffic_circuit',
                'features': {
                    'x1': current_segment_data[0].tolist(),
                    'x2': current_segment_data[1].tolist(),
                    'x3': current_segment_data[2].tolist(),
                    'x4': current_segment_data[3].tolist(),
                    'phase': current_segment_data[7].tolist() if len(current_segment_data) > 7 else []
                }
            }
            # Always update the labeled_data_list at current_index position
            if current_index < len(st.session_state.labeled_data_list):
                st.session_state.labeled_data_list[current_index] = entry
            else:
                st.session_state.labeled_data_list.append(entry)
            # Advance to next segment (optionally skipping already labeled)
            next_idx = min(len(segments_to_label)-1, current_index + 1)
            if st.session_state.skip_labeled and st.session_state.labeled_data_list:
                N = len(segments_to_label)
                i = next_idx
                while i < N:
                    lab = st.session_state.labeled_data_list[i].get('label') if i < len(st.session_state.labeled_data_list) else None
                    if lab is None or (pd.isna(lab)):
                        break
                    i += 1
                next_idx = min(N-1, i)
            st.session_state.current_index = next_idx
            st.rerun()
    
    with col2:
        if st.button("❌ No, Not a Circuit", use_container_width=True):
            seg_info = current_segment_info
            entry = {
                'unique_flight_id': seg_info.get('unique_flight_id'),
                'segment_num': seg_info.get('segment_num', current_index),
                'flight_pickle': seg_info.get('flight_pickle'),
                'airport_code': seg_info.get('airport_code'),
                'rwy': seg_info.get('rwy'),
                'segment_start_idx': seg_info.get('segment_start_idx'),
                'segment_end_idx': seg_info.get('segment_end_idx'),
                'label': 'not_a_circuit',
                'features': {
                    'x1': current_segment_data[0].tolist(),
                    'x2': current_segment_data[1].tolist(),
                    'x3': current_segment_data[2].tolist(),
                    'x4': current_segment_data[3].tolist(),
                    'phase': current_segment_data[7].tolist() if len(current_segment_data) > 7 else []
                }
            }
            # Always update the labeled_data_list at current_index position
            if current_index < len(st.session_state.labeled_data_list):
                st.session_state.labeled_data_list[current_index] = entry
            else:
                st.session_state.labeled_data_list.append(entry)
            # Advance to next segment (optionally skipping already labeled)
            next_idx = min(len(segments_to_label)-1, current_index + 1)
            if st.session_state.skip_labeled and st.session_state.labeled_data_list:
                N = len(segments_to_label)
                i = next_idx
                while i < N:
                    lab = st.session_state.labeled_data_list[i].get('label') if i < len(st.session_state.labeled_data_list) else None
                    if lab is None or (pd.isna(lab)):
                        break
                    i += 1
                next_idx = min(N-1, i)
            st.session_state.current_index = next_idx
            st.rerun()

    with col3:
        if st.button("⛔ Omit segment", use_container_width=True):
            # Mark omitted
            seg_info = current_segment_info
            omit_entry = {
                'unique_flight_id': seg_info.get('unique_flight_id'),
                'segment_num': seg_info.get('segment_num', current_index),
                'flight_pickle': seg_info.get('flight_pickle'),
                'airport_code': seg_info.get('airport_code'),
                'rwy': seg_info.get('rwy'),
                'segment_start_idx': seg_info.get('segment_start_idx'),
                'segment_end_idx': seg_info.get('segment_end_idx'),
                'label': 'omitted',
                'features': {}
            }
            # Update the labeled_data_list at current_index position
            if current_index < len(st.session_state.labeled_data_list):
                st.session_state.labeled_data_list[current_index] = omit_entry
            else:
                st.session_state.labeled_data_list.append(omit_entry)

            # Move to next segment (optionally skipping already labeled)
            next_idx = min(len(segments_to_label)-1, current_index + 1)
            if st.session_state.skip_labeled and st.session_state.labeled_data_list:
                N = len(segments_to_label)
                i = next_idx
                while i < N:
                    lab = st.session_state.labeled_data_list[i].get('label') if i < len(st.session_state.labeled_data_list) else None
                    if lab is None or (pd.isna(lab)):
                        break
                    i += 1
                next_idx = min(N-1, i)
            st.session_state.current_index = next_idx
            st.rerun()

    with col4:
        if st.button("↩️ Undo last", use_container_width=True):
            if st.session_state.labeled_data_list:
                st.session_state.labeled_data_list.pop()
                st.session_state.current_index = max(0, st.session_state.current_index - 1)
                st.rerun()
            else:
                st.warning("No labeled segments to undo.")

    # If we loaded an original DataFrame, allow exporting edits back into it
    if 'loaded_original_df' in st.session_state:
        if st.button("Export edits to new parquet", use_container_width=True):
            try:
                df_original = st.session_state.loaded_original_df.copy()
                edited = st.session_state.labeled_data_list
                # If we saved original indices mapping, use it to map edits back to exact rows
                if 'loaded_original_indices' in st.session_state and st.session_state.loaded_original_indices:
                    mapping = st.session_state.loaded_original_indices
                    for i, entry in enumerate(edited):
                        if i < len(mapping):
                            original_row = mapping[i]
                            if original_row < len(df_original):
                                df_original.at[original_row, 'label'] = entry.get('label')
                                df_original.at[original_row, 'features'] = entry.get('features')
                else:
                    # Fallback: map by row order
                    for i, entry in enumerate(edited):
                        if i < len(df_original):
                            df_original.at[i, 'label'] = entry.get('label')
                            df_original.at[i, 'features'] = entry.get('features')

                # For omitted entries, ensure they are marked in the original DF if indices available
                if 'omitted_entries' in st.session_state and st.session_state.omitted_entries and 'loaded_original_indices' in st.session_state:
                    # We don't have direct mapping for omitted entries beyond earlier removal, so just note omitted rows
                    df_original['omitted'] = False
                    for i, original_idx in enumerate(st.session_state.loaded_original_indices):
                        # If original index beyond current mapping, mark omitted
                        pass

                # Prepare in-memory parquet for download
                buffer = io.BytesIO()
                df_original.to_parquet(buffer, index=False)
                buffer.seek(0)
                st.download_button("Download updated labeled parquet", data=buffer.read(), file_name="labeled_ground_truth_updated.parquet", mime='application/octet-stream', key="download_updated")
            except Exception as e:
                logging.exception(e)
                st.error(f"Failed to export edited parquet: {e}")

if __name__ == "__main__":
    main()
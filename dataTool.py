#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import argparse
import subprocess
import fnmatch
import re

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process multiple data files and generate a summary report.")
parser.add_argument("base_dir", type=str,nargs="?",default=".", help="Base directory containing the case directories (default: current directory)")
parser.add_argument(
    "--exclude_dirs", "-e", nargs="*", default=[],
    help="List of directory name patterns (regex) to exclude"
)

args = parser.parse_args()

def _run_subprocess_command(command, script_path):
    """Helper function to run subprocess command."""
    try:
        print(f"Running command{command}")
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:\n{e.stderr}")
        return False

def natural_sort_key(s):
    """Natural sort key function for proper numerical sorting"""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', s)]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def walk_directory(base_dir, fi, exclude_patterns, max_search_depth):
    """Walk through the directory and find all 'fl.dat' files up to a specific depth."""
    found_objects = []
    base_depth = base_dir.rstrip(os.sep).count(os.sep)
    exclude_patterns = exclude_patterns or []
    exclude_regexes = [re.compile(p) for p in exclude_patterns]

    # Parse fi into filename and parent_dir_pattern (if any)
    filename_pattern = fi
    parent_dir_pattern = None
    if fi and os.sep in fi:
        # Split into directory and filename parts
        parent_dir_pattern, filename_pattern = os.path.split(fi)
        # Normalize to avoid leading/trailing slashes
        parent_dir_pattern = parent_dir_pattern.rstrip(os.sep)

    for root, dirs, files in os.walk(base_dir, topdown=True):
        current_depth = root.rstrip(os.sep).count(os.sep) - base_depth
        # Exclude directories matching any pattern
        dirs[:] = [
            d for d in dirs
            if not any(regex.search(d) for regex in exclude_regexes)
        ]
        # Exclude timesteps
        filtered_dirs = []
        for d in dirs:
            full_path = os.path.join(root, d)
            if is_number(d):
                try:
                    subdirs = os.listdir(full_path)
                except Exception:
                    subdirs = []
                if 'uniform' in subdirs:
                    continue  # Skip this number-named dir (it's a timestep)
            filtered_dirs.append(d)
        dirs[:] = filtered_dirs

        if current_depth >= max_search_depth:
            dirs[:] = []
        # print(f"Searching in directory: {root}")
        
        # Check if the current directory matches parent_dir_pattern (if specified)
        if parent_dir_pattern:
            rel_path = os.path.relpath(root, base_dir)
            if not rel_path.endswith(parent_dir_pattern):
                continue  # Skip if parent structure doesn't match

        match_found = False
        # Search for matching files/dirs
        for file in files:
            if filename_pattern and fnmatch.fnmatch(file, filename_pattern):
                full_path = os.path.join(root, file)
                found_objects.append(full_path)
                print(f"Found file: {full_path}")
                match_found = True

        if match_found:
            dirs[:] = []  # Prevent going deeper into subdirs
            continue  # Move to next directory

        for dir in dirs:
            if filename_pattern and fnmatch.fnmatch(dir, filename_pattern):
                full_path = os.path.join(root, dir)
                found_objects.append(full_path)
                print(f"Found directory: {full_path}")
                match_found = True

        if match_found:
            dirs[:] = []  # Prevent going deeper into subdirs

    return found_objects

def extract_names(found_objects, sep):
    """Extract all parts from the directory path, optionally splitting each part by a separator."""
    extracted_names = {}

    for obj in found_objects:
        parts = obj.split(os.sep)
        split_parts = []
        for part in parts:
            if sep is not None and sep in part:
                split_parts.extend(part.split(sep))
            else:
                split_parts.append(part)
        extracted_names[obj] = split_parts
        print(f"Extracted names from '{obj}': {extracted_names[obj]}")
    return extracted_names

def calcAmplitude(found_objs_with_names, casename1_Index, casename2_Index, freq_Index, selected_columns, delimiter, skiprows, output):

    results = {}
    for obj, name_dicts in found_objs_with_names.items():
        try:
            # Use any whitespace as the delimiter
            data = np.loadtxt(obj, delimiter=delimiter, skiprows=skiprows)
        except IOError:
            raise IOError(f"Could not read data file '{obj}'")

        # Extract time column (always column 0)
        time = data[:, 0]
        
        # Determine which columns to process
        if selected_columns is None:
            # Use all columns except time (column 0)
            num_columns = data.shape[1]
            selected_columns = list(range(1, num_columns))  # Skip column 0 (time)
        
        # Process each selected column
        case_name = f"{name_dicts[casename1_Index]}_{name_dicts[casename2_Index]}"
        frequency = float(name_dicts[freq_Index])
        T = 1 / frequency
        print(f"Processing file '{obj}' with frequency {frequency} Hz")
        
        # Initialize storage for all columns
        if case_name not in results:
            results[case_name] = {
                "frequencies": [],
                "column_data": {}  # Will store data for each column
            }
        
        # Store frequency (same for all columns)
        results[case_name]["frequencies"].append(frequency)
        
        # Process each selected column
        for col_idx in selected_columns:
            values = data[:, col_idx]
            
            # Initialize storage for this column if not exists
            if col_idx not in results[case_name]["column_data"]:
                results[case_name]["column_data"][col_idx] = {
                    "periods_amp": [],
                    "amplitude": []
                }
            
            # Divide data into periods of approximately T length
            period_amplitude = []
            current_period_start = time[0]
            current_period_values = []

            for t, value in zip(time, values):
                if t >= current_period_start + T:
                    # Calculate max-min for the completed period
                    if current_period_values:
                        period_max = max(current_period_values)
                        period_min = min(current_period_values)
                        period_amplitude.append(period_max - period_min)
                    # Start a new period
                    current_period_start += T
                    current_period_values = []

                # Add the value to the current period
                current_period_values.append(value)

            # Handle the last (possibly incomplete) period
            if current_period_values:
                last_period_duration = time[-1] - current_period_start
                if last_period_duration >= 0.95 * T:
                    period_max = max(current_period_values)
                    period_min = min(current_period_values)
                    period_amplitude.append(period_max - period_min)

            if not period_amplitude:
                raise ValueError(f"No complete periods found in file '{obj}' column {col_idx}")

            # Calculate second half average
            amp = (
                np.mean(period_amplitude[len(period_amplitude) // 2:])
                if len(period_amplitude) > 1
                else period_amplitude[0]
            )
            
            # Store results for this column
            results[case_name]["column_data"][col_idx]["periods_amp"].append(period_amplitude)
            results[case_name]["column_data"][col_idx]["amplitude"].append(amp)

    # Write amplitude values to the first output file
    with open(f"{output}_amplitude", "w") as ampFile:
        for case_name, data in results.items():
            # Sort the data by frequency
            sorted_cols = sorted(data["column_data"].keys())
            sorted_data = sorted(
                zip(data["frequencies"],
                    *[data["column_data"][col_idx]["amplitude"] for col_idx in sorted_cols]),
                key=lambda x: x[0]  # Sort by frequency
            )
            
            # Write case name header
            ampFile.write(f"{case_name}\n")
            
            # Write column headers
            header = "Frequency\t" + "\t".join(f"Col{col_idx}_amp" for col_idx in sorted_cols)
            ampFile.write(header + "\n")
            
            # Write data for each frequency
            for row in sorted_data:
                freq = row[0]
                amp_values = row[1:]
                line = f"{freq:.3f}\t" + "\t".join(f"{val:.3f}" for val in amp_values)
                ampFile.write(line + "\n")
            
            ampFile.write("\n")
    print(f"Wrote amplitude to {output}_amplitude")

    # Write all periods amplitude to the second output file
    with open(f"{output}_periods_amplitude", "w") as p_ampFile:
        for case_name, data in results.items():
            # Sort the data by frequency and columns
            sorted_cols = sorted(data["column_data"].keys())
            sorted_data = sorted(
                zip(data["frequencies"],
                    *[data["column_data"][col_idx]["periods_amp"] for col_idx in sorted_cols]),
                key=lambda x: x[0]  # Sort by frequency
            )
            
            # Write case name header
            p_ampFile.write(f"{case_name}\n")
            
            # Write data for each column separately
            for col_idx in sorted_cols:
                p_ampFile.write(f"Column {col_idx}\n")
                p_ampFile.write("Frequency\teach period's amplitude\n")
                
                for row in sorted_data:
                    freq = row[0]
                    periods_amp = row[1 + sorted_cols.index(col_idx)]
                    periods_amp_str = "\t".join(f"{val:.3f}" for val in periods_amp)
                    p_ampFile.write(f"{freq:.3f}\t{periods_amp_str}\n")
                
                p_ampFile.write("\n")
    print(f"Wrote amplitude of rach period to {output}_periods_amplitude")

def PnumaticPower(found_objs_with_names, casename_Index, freq_Index, flux_delimiter, flux_col_Index, pressure_delimiter, pressure_col_Index, output_file):
    results = {}
    for obj, name_dicts in found_objs_with_names.items():
        try:
            # Load flux data using np.genfromtxt for better error handling
            flux_data = np.genfromtxt(obj, delimiter=flux_delimiter, invalid_raise=False, comments="#")
        except IOError:
            raise IOError(f"Could not read flux data file '{obj}'")

        # Construct the relative path for the pressure file
        pressure_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(obj))), "patchProbes", "0", "p")
        try:
            # Load pressure data using np.genfromtxt for better error handling
            pressure_data = np.genfromtxt(pressure_file, delimiter=pressure_delimiter, invalid_raise=False, comments="#")
        except IOError:
            raise IOError(f"Could not read pressure data file '{pressure_file}'")

        # Extract time and flux values
        flux_time = flux_data[:, 0]
        flux_values = flux_data[:, flux_col_Index]

        # Extract time and pressure values
        pressure_time = pressure_data[:, 0]
        file_number = int(re.search(r'(\d+)', os.path.basename(obj)).group(1)) if re.search(r'(\d+)', os.path.basename(obj)) else None
        pressure_values = pressure_data[:, pressure_col_Index]
        # pressure_values = pressure_data[:, (file_number*2)-1]

        if not len(flux_time) == len(pressure_time):
            print(f"Time values do not match in '{os.sep.join(os.path.dirname(obj).split(os.sep)[:3])}'. Interpolating to match lengths.")
            # Interpolate the longer array onto the shorter one's time points
            if len(flux_time) > len(pressure_time):
                # Interpolate flux onto pressure_time
                flux_values = np.interp(pressure_time, flux_time, flux_values)
                flux_time = pressure_time
            else:
                # Interpolate pressure onto flux_time
                pressure_values = np.interp(flux_time, pressure_time, pressure_values)
                pressure_time = flux_time

        # Calculate power for each time step
        # power_values = np.abs(flux_values) * (pressure_values-100000)
        power_values = flux_values * (100000 - pressure_values)
        
        # Calculate average power for the second half of the results (time-weighted)
        second_half_time = flux_time[len(flux_time) // 2:]
        second_half_power = power_values[len(power_values) // 2:]
        avg_power = np.trapz(second_half_power, second_half_time) / (second_half_time[-1] - second_half_time[0])
        # avg_power = np.mean(second_half_power)
       

        # Extract case name and frequency
        case_name = name_dicts[casename_Index]
        frequency = float(name_dicts[freq_Index])

        if case_name not in results:
            results[case_name] = {}
        if frequency not in results[case_name]:
            results[case_name][frequency] = {"frequencies": [], "avg_power": [], "power_values": [], "file_number": [], "time_arrays": None}
        results[case_name][frequency]["avg_power"].append(avg_power)
        results[case_name][frequency]["power_values"].append(power_values)
        results[case_name][frequency]["file_number"].append(file_number)
        results[case_name][frequency]["time_array"] = flux_time


    for case_name, freq_dict in results.items():
        outfilename = f"{case_name}_file.txt"
        # Prepare columns grouped by frequency
        freq_columns = []
        max_length = 0
        for freq, v in sorted(freq_dict.items()):
            time_arr = v["time_array"]
            value_columns = []
            for fn, arr in zip(v["file_number"], v["power_values"]):
                value_columns.append((fn, arr))
                max_length = max(max_length, len(arr))
            max_length = max(max_length, len(time_arr))
            freq_columns.append((freq, time_arr, value_columns))

        with open(outfilename, "w") as f:
            header = []
            # Build header: time_{freq} then all value columns for that freq
            for freq, _, value_columns in freq_columns:
                header.append(f"time_{freq:.3f}")
                for fn, _ in value_columns:
                    header.append(f"{freq:.3f}[{fn}]")
            f.write("\t".join(header) + "\n")
            # Write data rows
            for i in range(max_length):
                row = []
                for _, time_arr, value_columns in freq_columns:
                    row.append(f"{time_arr[i]:.6f}" if i < len(time_arr) else "")
                    for _, arr in value_columns:
                        row.append(f"{arr[i]:.3f}" if i < len(arr) else "")
                f.write("\t".join(row) + "\n")

    # --- 2. Summary file ---
    summary_filename = "summary_all_cases.txt"
    for case_name, freq_dict in results.items():
        summary = {}
        for freq, v in freq_dict.items():
            if freq not in summary:
                summary[freq] = []
            summary[freq].extend(zip(v["file_number"], v["avg_power"]))

        if summary:
            max_pairs = max(len(v) for v in summary.values())
            with open(summary_filename, "a") as f:  # Append mode
                # Write case name as a header
                f.write(f"Case: {case_name}\n")
                header = ["Frequency", "SumAvgPower"]
                all_file_numbers = set()
                for pairs in summary.values():
                    all_file_numbers.update(fn for fn, _ in pairs)
                all_file_numbers = sorted(all_file_numbers)
                for fn in all_file_numbers:
                    header.append(f"avg_power[{fn}]")
                f.write("\t".join(header) + "\n")
                for freq in sorted(summary.keys()):
                    pairs = summary[freq]
                    fn_to_avg = {fn: ap for fn, ap in pairs}
                    sum_avg_power = sum(fn_to_avg.values())
                    row = [f"{freq:.3f}", f"{sum_avg_power:.6f}"]
                    for fn in all_file_numbers:
                        row.append(f"{fn_to_avg[fn]:.6f}" if fn in fn_to_avg else "")
                    f.write("\t".join(row) + "\n")
                f.write("\n")  # Separate cases with a blank line
        else:
            print(f"No valid data found for summary file for case {case_name}. Skipping.")

    
def run_plotTools(found_objs_with_names, casename1_Index, casename2_Index, plot_type, batch=False):
    """Run an external Python script for each object."""

    # Construct the command to run the external script
    script_path = os.path.expanduser("~/plotTool/plotTools.py")  # Use the correct path to the script

    if batch:
        # Batch mode command
        command = ["python3", script_path, "-pt", plot_type, "-sd"]
        for obj, name_dicts in found_objs_with_names.items():
            command.extend([obj, "-label", f"{name_dicts[casename1_Index]}_{name_dicts[casename2_Index]}"])
        _run_subprocess_command(command, script_path)
    else:
        # Individual mode
        for obj, name_dicts in found_objs_with_names.items():
            command = [
                "python3", script_path, "-pt", plot_type, "-sd",
                "-ft", f"{name_dicts[casename1_Index]}_{name_dicts[casename2_Index]}",
                obj, "-label", f"{name_dicts[casename1_Index]}_{name_dicts[casename2_Index]}"
            ]
            _run_subprocess_command(command, script_path)


def integrate_timeseries(found_objs_with_names, casename1_Index, casename2_Index, write_file_Index, output, output_format, delimiter, skiprows, selected_columns):
    """
    Reads flux timeseries files and combines them into either CSV files or an Excel file.
    """
    usecols = None
    if selected_columns is not None:
        usecols = [0] + [c for c in selected_columns if 0 < c]

    write_data = {}

    def _process_case_data(cases_data):
            all_series = []
            column_names = []

            for case_key in sorted(cases_data.keys(), key=natural_sort_key):
                files = cases_data[case_key]
                num_files = len(files)
                
                # Add time series (column 0)
                file_name, data = next(iter(files.items()))
                all_series.append(data.iloc[:, 0].values)
                column_names.append(f"{case_key}_Time")

                # Add all remaining columns (already filtered during read)
                for file_name, data in sorted(files.items(), key=lambda x: natural_sort_key(x[0])):
                    num_columns = data.shape[1]
                    for idx in range(1, num_columns):  # Now all columns are ones we want
                            all_series.append(data.iloc[:, idx].values)
                            suffix = "" if num_columns == 2 else f"_Val{idx}"
                            column_name = f"{case_key}{suffix}" if num_files == 1 else f"{case_key}_{file_name}{suffix}"
                            column_names.append(column_name)

            # Pad series to equal length
            max_len = max(len(s) for s in all_series)
            return pd.DataFrame({
                name: np.pad(s, (0, max_len - len(s)), mode='constant', constant_values=np.nan)
                if len(s) < max_len else s
                for name, s in zip(column_names, all_series)
            })

    # Read all data
    print(f"Processing files ... ")
    for obj, name_dicts in found_objs_with_names.items():
        try:
            case_key = f"{name_dicts[casename1_Index]}_{name_dicts[casename2_Index]}"
            write_key = name_dicts[write_file_Index]
            file_name = os.path.splitext(os.path.basename(obj))[0]  # Remove extension
            
            data = pd.read_csv(obj, sep=delimiter, header=skiprows, comment="#", usecols=usecols)

            if data.empty:
                raise ValueError(f"File contains no data after processing headers/skiprows")

            write_data.setdefault(write_key, {}).setdefault(case_key, {})[file_name] = data
        
        except Exception as e:
            error_msg = f"Error processing file '{obj}': {str(e)}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg) from e

    # Process data for each write key
    processed = {
        tab: _process_case_data(cases_data)
        for tab, cases_data in write_data.items()
    }

    # Generate output
    if output_format.lower() == 'excel':
        with pd.ExcelWriter(f"{output}.xlsx", engine='xlsxwriter') as writer:
            for tab, df in sorted(processed.items(), key=lambda x: natural_sort_key(x[0])):
                df.to_excel(writer, sheet_name=tab, index=False)
        print(f"Created Excel file: {output}.xlsx")

    elif output_format.lower() == 'csv':
        os.makedirs(output, exist_ok=True)
        for tab, df in sorted(processed.items(), key=lambda x: natural_sort_key(x[0])):
            df.to_csv(os.path.join(output, f"timeseries_{tab}.csv"), index=False)
        print(f"Created CSV files in: {output}")

    else:
        raise ValueError("Output format must be 'csv' or 'excel'")

def main(args):
    
    input_file = "*Variable_2.csv"     # "log.compressibleInterDyMFoam" "patchProbes/0/p" "fl[0-9]*" "height.dat" "*Variable_2.csv"
    output = "roll"

    exclude_patterns = ['^processor', 'dynamicCode', 'polyMesh']

    found_objs = walk_directory(args.base_dir, input_file, exclude_patterns + args.exclude_dirs, max_search_depth=8)
    found_objs_with_names = extract_names(found_objs, sep='_')
    
    # Ask the user which functions to run
    print("\nAvailable processing functions:")
    print("1. calcAmplitude")
    print("2. PnumaticPower")
    print("3. run_plotTools (motion)")
    print("4. integrate_timeseries")
    choice = input("\nEnter your choice (or 'q' to quit): ").strip().lower()
    # choice = 4

    for i in range(1):
        
        if choice == '1':
            calcAmplitude(found_objs_with_names, casename1_Index=-6, casename2_Index=-4, freq_Index=-3, selected_columns=[1], delimiter=',', skiprows=1, output=output)

        elif choice == '2':
            PnumaticPower(found_objs_with_names, casename_Index=-6, freq_Index=-5, flux_delimiter='\t', flux_col_Index=3, pressure_delimiter=None, pressure_col_Index=6, output_file=output)

        elif choice == '3':
            run_plotTools(found_objs_with_names, casename1_Index=-3, casename2_Index=-2, plot_type="motion", batch=False)

        elif choice == '4':
            integrate_timeseries (found_objs_with_names, casename1_Index=-6, casename2_Index=-7, write_file_Index=-5, output=output, output_format='excel', delimiter="\s+", skiprows=0, selected_columns=[3])  # delimiter=r"\s+"


if __name__ == "__main__":
    main(args)

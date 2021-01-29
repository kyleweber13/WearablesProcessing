import gt3x
import pyedflib
from datetime import datetime
import matplotlib.pyplot as plt
# Requires bitsring package


def import_gt3x(filepath):
    """Imports timestamps and accelerometer channel from GT3X file."""

    print("\nImporting data from {}...".format(filepath))

    accel, ts, meta_data = gt3x.read_gt3x(filepath)

    start_stamp = str(ts[0])
    stop_stamp = str(ts[-1])
    tz = meta_data["TimeZone"]
    sample_rate = meta_data["Sample_Rate"]
    acc_min = meta_data["Acceleration_Min"]
    acc_max = meta_data["Acceleration_Max"]

    header_info = {"Start_time": start_stamp, "Stop_time": stop_stamp, "Timezone": tz, "Sample_rate": int(sample_rate),
                   "Accel_range": (float(acc_min), float(acc_max))}

    print("Complete.")

    return header_info, accel


def export_edf(filepath, output_dir):
    """Calls import_gt3x and converts data to EDf format.

    :argument
    -filepath: full filepath to gt3x file
    -output_dir: pathway where EDF file gets written
    """

    metadata, acc_data = import_gt3x(filepath=filepath)

    channel_names = ["Acc_x", "Acc_y", "Acc_z"]

    signal_headers = pyedflib.highlevel.make_signal_headers(channel_names, sample_rate=metadata["Sample_rate"])
    header = pyedflib.highlevel.make_header(startdate=datetime.strptime(metadata["Start_time"],
                                                                        "%Y-%m-%dT%H:%M:%S.%f"))
    output_filedir = output_dir + filepath.split("/")[-1].split(".")[0] + ".EDF"
    pyedflib.highlevel.write_edf(output_filedir, acc_data.transpose(), signal_headers, header)

    print("File saved as {}.".format(output_filedir))


# export_edf(filepath='C:/Users/ksweber/Desktop/TAS1H19200131.gt3x', output_dir="C:/Users/ksweber/Desktop/")


import os
import numpy as np
import librosa

def analyze_wav_durations(directory):
    """
    Reads all .wav files from the given directory and returns
    average, min, max, and std deviation of durations (in seconds), 
    as well as the index of the file with min and max duration,
    and also counts number of files in given ranges:
      1-2, 2-3, ..., 9-10 seconds, and >10 seconds.

    Args:
        directory (str): Path to directory containing .wav files.

    Returns:
        dict with basic stats and a key 'counts_by_range' mapping str(range) to count, e.g.:
        {
            'mean': ...,
            'min': ...,
            'max': ...,
            'std': ...,
            'min_index': ...,
            'max_index': ...,
            'min_fname': ...,
            'max_fname': ...,
            'counts_by_range': {
                '1-2': ...,
                '2-3': ...,
                ...
                '9-10': ...,
                '>10': ...,
            },
        }
    """
    durations = []
    fnames_list = []
    for fname in os.listdir(directory):
        if fname.lower().endswith('.wav'):
            fpath = os.path.join(directory, fname)
            try:
                # Only load header for duration
                duration = librosa.get_duration(filename=fpath)
                durations.append(duration)
                fnames_list.append(fname)
            except Exception as e:
                print(f"Warning: Could not process {fpath}: {e}")
    if not durations:
        raise ValueError(f"No .wav files found in directory {directory}")
    durations = np.array(durations)
    min_index = int(np.argmin(durations))
    max_index = int(np.argmax(durations))

    # Compute counts per range
    counts_by_range = {}
    bins = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10)]
    for (start, end) in bins:
        count = int(np.sum((durations >= start) & (durations < end)))
        counts_by_range[f"{start}-{end}"] = count
    # Count for >10s
    counts_by_range[">10"] = int(np.sum(durations > 10))

    return {
        'mean': float(np.mean(durations)),
        'min': float(np.min(durations)),
        'max': float(np.max(durations)),
        'std': float(np.std(durations)),
        'min_index': min_index,
        'max_index': max_index,
        'min_fname': fnames_list[min_index],
        'max_fname': fnames_list[max_index],
        'counts_by_range': counts_by_range,
    }

result = analyze_wav_durations("raw_audio")
print(result)
print("Counts by duration range:")
for k, v in result['counts_by_range'].items():
    print(f"  {k} seconds: {v} files")
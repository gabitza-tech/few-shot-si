import sys
import os
from pydub import AudioSegment
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Function to convert .flac to .wav
def convert_flac_to_wav(file_path):
    try:
        # Load the .flac file
        audio = AudioSegment.from_file(file_path, format="flac")

        # Replace .flac with .wav
        wav_path = file_path.replace(".flac", ".wav")

        # Export the file as .wav
        audio.export(wav_path, format="wav")
        return f"Converted: {file_path} to {wav_path}"
    except Exception as e:
        return f"Failed to convert {file_path}: {str(e)}"

# Wrapper function to use with Pool and tqdm
def convert_wrapper(args):
    return convert_flac_to_wav(*args)

# Function to traverse directories and convert files using multiprocessing
def convert_all_flac_in_directory(directory):
    # List to hold all the .flac file paths
    flac_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".flac"):
                file_path = os.path.join(root, file)
                flac_files.append((file_path,))

    # Use all available CPUs for processing
    num_workers = cpu_count()

    # Set up the multiprocessing pool
    with Pool(num_workers) as pool:
        # Use tqdm to display the progress bar
        results = list(tqdm(pool.imap(convert_wrapper, flac_files), total=len(flac_files)))

    # Print the results
    for result in results:
        print(result)

if __name__ == "__main__":
    # Specify the root directory containing the class folders
    root_directory = sys.argv[1]

    # Convert all .flac files in the directory
    convert_all_flac_in_directory(root_directory)


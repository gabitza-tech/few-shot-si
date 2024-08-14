import os
import json
import librosa
import argparse
import logging
import soundfile as sf
import random
import librosa
from pydub import AudioSegment
import soundfile as sf
from multiprocessing import Pool
import time
import numpy as np

"""
We take a random crop of size x seconds!

-i/--input_dir must have the following structure:

DATASET
--class
----fileid_1.wav
----fileid_2.wav
...

Output directory will have the following structure:
DATASET_dur_{x}_ovl_{y}_min_{z}
--class
----fileid_1_start_{s1}.wav
----fileid_1_start_{s2}.wav
...
"""

def parse_arguments():
    parser = argparse.ArgumentParser(description="Segment audio files")
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Input directory containing audio files")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Output directory to save segmented audio files")
    parser.add_argument("-dur", "--crop_duration", type=float, default = 3,
                        help="Duration of each segment in seconds")
    parser.add_argument("-min", "--minimum_length", type=float, default=0,
                        help="Minimum duration of an utterance from the database, we ignore/discard utterances that are less than that length. (we do not split them)")
    
    return parser.parse_args()

def crop_audio(input_audio_file, output_audio_file, target_duration=3):
    # Load audio using librosa
    audio, sr = librosa.load(input_audio_file, sr=None)
    # Get duration of the audio in seconds
    duration = librosa.get_duration(y=audio, sr=sr)
    
    # If audio duration is longer than target duration, randomly select a segment
    if duration > target_duration:
        # Calculate start time for cropping
        start_time = random.uniform(0, duration - target_duration)
        
        # Crop the audio
        cropped_audio = audio[int(start_time * sr):int((start_time + target_duration) * sr)]
    else:
        # If audio duration is shorter, repeat and clip until it reaches target duration
        # Calculate the number of repetitions needed to reach or exceed the target duration
        num_repetitions = int(target_duration / duration) + 1

        repeated_audio = audio.copy()

        # Repeat the audio by concatenating it with itself
        for _ in range(num_repetitions - 1):
            repeated_audio = np.concatenate([repeated_audio, audio.copy()])

        # Trim the repeated audio to match the target duration
        cropped_audio = repeated_audio[:int(sr * target_duration)]

    # Save cropped audio
    sf.write(output_audio_file, cropped_audio, sr)

def generate_crop(input_dir, output_dir, crop_dur, min_utter):
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    
    verify_dir = any(os.path.isdir(os.path.join(input_dir, item)) for item in os.listdir(input_dir))
    if verify_dir:
        for dir in os.listdir(input_dir):
            if "._" in dir:
                continue

            in_utt_dir = os.path.join(input_dir,dir)
            out_utt_dir = os.path.join(output_dir,dir)

            if not os.path.exists(out_utt_dir):
                os.mkdir(out_utt_dir)
            
            for file in os.listdir(in_utt_dir):
                if "._" in file or ".mkv" in file:
                    continue
                file_in = os.path.join(in_utt_dir,file)
                
                if ".flac" in file:
                    file = file.replace(".flac",".wav")

                file_out = os.path.join(out_utt_dir,file)
                crop_audio(file_in, file_out, target_duration=crop_dur)
    else:
        for file in os.listdir(input_dir):
            file_in = os.path.join(input_dir,file)
            file_out = os.path.join(output_dir,file)
            crop_audio(file_in, file_out, target_duration=crop_dur)
            
def process_speaker(input_output_dirs):
    input_dir, output_dir,crop_dur,min_utter = input_output_dirs
    generate_crop(input_dir, output_dir, crop_dur, min_utter)


if __name__ == "__main__":
    start = time.time()
    random.seed(42)
    args = parse_arguments()

    input_dir = args.input_dir
    output_dir = args.output_dir
    crop_dur = args.crop_duration
    min_utter = args.minimum_length


    speaker_classes = os.listdir(input_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    """
    for spk in sorted(speaker_classes):
        in_spk_dir = os.path.join(input_dir,spk)
        out_spk_dir = os.path.join(output_dir, spk)
        print(spk)
        generate_crop(in_spk_dir, out_spk_dir, crop_dur, min_utter)
    """

    pool = Pool()  # Initialize the Pool for multiprocessing

    input_output_dirs = [(os.path.join(input_dir, spk), os.path.join(output_dir, spk),crop_dur,min_utter) for spk in sorted(speaker_classes)]
    pool.map(process_speaker, input_output_dirs)  # Map the function to process each speaker to the Pool
    pool.close()
    pool.join()  # Wait for all processes to finish before continuing
    

    dur = time.time() - start
    print(f"Time taken to crop all audios is {dur} seconds")

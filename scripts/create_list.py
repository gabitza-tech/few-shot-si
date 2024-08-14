import os
import sys

"""
Get file list like:
id filepath

Used for extracting embeddings later!
"""

def generate_file_list(root_directory, output_file, class_list):
    with open(output_file, 'w') as f:
        # Initialize class ID from folder names
        for class_folder in sorted(os.listdir(root_directory)):
            if class_folder not in class_list:
                continue

            class_folder_path = os.path.join(root_directory, class_folder)
            
            if os.path.isdir(class_folder_path):
                # Iterate through items in the class folder
                for item in sorted(os.listdir(class_folder_path)):
                    item_path = os.path.join(class_folder_path, item)
                    
                    if os.path.isdir(item_path):
                        # Iterate through WAV files in subfolders
                        for wav_file in sorted(os.listdir(item_path)):
                            if wav_file.endswith('.wav'):
                                f.write(f"{class_folder} {os.path.join(item_path, wav_file)}\n")
                    elif item.endswith('.wav'):
                        # Process WAV files directly in the class folder
                        f.write(f"{class_folder} {item_path}\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_file_list.py root_directory out_file classes_list")
        sys.exit(1)

    root_directory = sys.argv[1]

    if not os.path.isdir(root_directory):
        print(f"Directory {root_directory} does not exist.")
        sys.exit(1)

    output_file = sys.argv[2]
    class_list = []
    with open(sys.argv[3],'r') as f:
        for line in f:
            class_list.append(line.strip())

    generate_file_list(root_directory, output_file, class_list)
    print(f"File list generated: {output_file}")


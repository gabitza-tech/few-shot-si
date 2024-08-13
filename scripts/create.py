import os
import sys

def generate_file_list(root_directory, output_file):
    with open(output_file, 'w') as f:
        # Initialize class ID from folder names
        for class_folder in sorted(os.listdir(root_directory)):
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
    if len(sys.argv) != 2:
        print("Usage: python generate_file_list.py root_directory")
        sys.exit(1)

    root_directory = sys.argv[1]

    if not os.path.isdir(root_directory):
        print(f"Directory {root_directory} does not exist.")
        sys.exit(1)

    output_file = 'file_list.txt'
    generate_file_list(root_directory, output_file)
    print(f"File list generated: {output_file}")


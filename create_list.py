import os
import sys
# Define the root directory containing the class subfolders
root_directory = sys.argv[1]

# Open a file to write the output
with open("output.txt", "w") as file:
    # Loop through each subfolder (class)
    for class_folder in os.listdir(root_directory):
        class_path = os.path.join(root_directory, class_folder)
        
        # Ensure it's a directory (skip files)
        if os.path.isdir(class_path):
            # Loop through all files in the class directory
            for filename in os.listdir(class_path):
                # Check if the file is a .wav file
                if filename.endswith(".wav"):
                    # Construct the full path
                    full_path = os.path.join(class_path, filename)
                    # Write the class name and full path to the file
                    file.write(f"{class_folder}\t{full_path}\n")


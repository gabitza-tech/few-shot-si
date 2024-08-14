import random
import sys

def split_file(input_file, output_file_10, output_file_90):
    # Read all lines from the input file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Shuffle the lines randomly
    random.shuffle(lines)

    # Calculate the split index
    total_lines = len(lines)
    split_index = int(total_lines * 0.1)

    # Write 10% of the lines to output_file_10
    with open(output_file_10, 'w') as file:
        file.writelines(lines[:split_index])

    # Write the remaining 90% of the lines to output_file_90
    with open(output_file_90, 'w') as file:
        file.writelines(lines[split_index:])

    print(f"Files split: {output_file_10} (10%), {output_file_90} (90%)")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python split_file.py input_file output_file_10 output_file_90")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file_10 = sys.argv[2]
    output_file_90 = sys.argv[3]

    split_file(input_file, output_file_10, output_file_90)

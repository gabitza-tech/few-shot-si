import os
import json
from collections import defaultdict
import sys

# Directory containing JSON files
directory = sys.argv[1]

def average_over_files(directory):
    # Initialize dictionaries to store sums and counts
    sum_values = defaultdict(lambda: defaultdict(float))
    count_values = defaultdict(lambda: defaultdict(int))
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                
                # Process TIM
                for key, value in data.get("tim", {}).items():
                    sum_values["tim"][key] += value
                    count_values["tim"][key] += 1
                
                # Process LaplacianShot
                for key, value in data.get("laplacianshot", {}).items():
                    sum_values["laplacianshot"][key] += value
                    count_values["laplacianshot"][key] += 1

    # Calculate averages
    averages = {}
    for method in sum_values:
        averages[method] = {
            key: sum_values[method][key] / count_values[method][key]
            for key in sum_values[method]
        }

    return averages

# Call the function and print results
averages = average_over_files(directory)
print(json.dumps(averages, indent=4))

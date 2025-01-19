import os
import csv
from datasets import Dataset, DatasetDict

# Configuration
logins_folder = "../logins/"
exclude_files =list(set(["178_144_125_23.csv",  "185_152_64_228.csv",  "83_9_37_128.csv",  "192_0_2_90.csv", "203_0_113_190.csv", "192_0_2_60.csv", "198_51_100_110.csv", "198_51_100_140.csv", "192_0_2_120.csv", "203_0_113_160.csv", "198_51_100_200.csv", "185_152_64_228.csv", "192_0_2_150.csv", "203_0_113_130.csv", "203_0_113_40.csv", "203_0_113_220.csv", "192_0_2_10.csv", "192_0_2_180.csv", "192_0_2_240.csv", "83_9_37_128.csv", "203_0_113_100.csv", "192_0_2_210.csv", "198_51_100_50.csv", "203_0_113_30.csv", "198_51_100_80.csv", "178_144_125_23.csv", "198_51_100_170.csv", "198_51_100_20.csv", "198_51_100_230.csv", "203_0_113_70.csv"]))

output_path = "merged_dataset"

def collect_exluded_files(folder,include):
    """Collect files from a folder excluding specified files."""
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f in include
    ]
def collect_files(folder, exclude):
    """Collect files from a folder excluding specified files."""
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f not in exclude
    ]

def read_csv_file(filepath):
    """Read a CSV file and return its content without the header."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            return [row for row in reader]
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    # Step 1: Collect all files from the folder
    files = collect_files(logins_folder, exclude_files)
    print(f"Found {len(files)} files to process.")

    # Step 2: Read and process data
    data = []
    # the good user data
    for file in files:
        content = read_csv_file(file)
        if content:
            for row in content:
                if len(row) > 0:
                    data.append({"instruction": row[0], "response": "OK"})
    # the bad user data

    bad_files = collect_exluded_files(logins_folder,exclude_files)
    for file in bad_files:
        content = read_csv_file(file)
        if content:
            for row in content:
                if len(row) > 0:
                    data.append({"instruction": row[0], "response": "BAD"})
    print(f"Aggregated {len(data)} records.")

    # Step 3: Create a Hugging Face Dataset
    dataset = Dataset.from_list(data)

    # Step 4: Save the dataset
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    main()


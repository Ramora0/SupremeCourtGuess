import os
import re


def rename_files_remove_prefix():
    """Remove 'https:--apps.oyez.org-player-#-' prefix from filenames in /data/raw_cases"""

    # Define the directory path
    data_dir = "./data/raw_cases/"

    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist!")
        return

    # Pattern to match the prefix (# represents any number)
    prefix_pattern = r"^https:--apps\.oyez\.org-player-#-"

    renamed_count = 0

    # Iterate through all files in the directory
    for filename in os.listdir(data_dir):
        old_path = os.path.join(data_dir, filename)

        # Skip if it's not a file
        if not os.path.isfile(old_path):
            continue

        # Check if filename matches the pattern
        if re.match(prefix_pattern, filename):
            # Remove the prefix
            new_filename = re.sub(prefix_pattern, "", filename)
            new_path = os.path.join(data_dir, new_filename)

            try:
                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
                renamed_count += 1
            except OSError as e:
                print(f"Error renaming {filename}: {e}")

    print(f"\nTotal files renamed: {renamed_count}")


if __name__ == "__main__":
    rename_files_remove_prefix()

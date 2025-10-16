import re
import os

# The directory containing log folders starting with '0516'
log_directory = 'saved/log/AA'

# Function to search directories and find info.log
def find_info_log_files():
    for root, dirs, files in os.walk(log_directory):
        for directory in dirs:
            if directory.startswith('0517'):
                yield os.path.join(root, directory, 'info.log'), directory

# Function to extract epoch information from log files
def extract_info(log_file_path, last_epoch_minus):
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    # Extracting the last epoch reported in the log file
    last_epoch_matches = list(re.finditer(r'epoch\s+:\s+(\d+)', log_content))
    if not last_epoch_matches:
        return None, None
    last_epoch = int(last_epoch_matches[-1].group(1))

    # Calculating the target epoch
    target_epoch = last_epoch - last_epoch_minus

    # Extracting information related to the target epoch
    target_epoch_info_pattern = rf'epoch\s+:\s+{target_epoch}(.+?)(?=epoch\s+:\s+|\Z)'
    target_epoch_info_match = re.search(target_epoch_info_pattern, log_content, re.DOTALL)

    return target_epoch_info_match.group(0).strip() if target_epoch_info_match else "Not Found", last_epoch

# Main script execution
if __name__ == "__main__":
    for log_file, directory in find_info_log_files():
        info, last_epoch = extract_info(log_file, 21)
        if info and last_epoch:
            print(f"Directory: {directory}")
            print(f"Extracted Information for Epoch {last_epoch - 21}:")
            print(info)
            print("--------------------------------------------------")
        else:
            print(f"Information for epoch {last_epoch - 21} not found in {log_file}")
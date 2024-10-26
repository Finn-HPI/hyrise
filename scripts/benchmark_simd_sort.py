import os
import sys
import subprocess
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def create_and_build(directory, system_name, l2_cache_size, build_mode):
    print("Build benchmark.")
       # Define the CMake command as a single string based on the system name
    if system_name == "AVX2":
        cmake_command = (
            'cmake -DCMAKE_C_COMPILER=/usr/bin/clang '
            '-DCMAKE_CXX_COMPILER=/usr/bin/clang++ '
            f'-GNinja -DCMAKE_BUILD_TYPE={build_mode} '
            '-DCMAKE_CXX_FLAGS="-Wno-switch-default -Wno-error=switch-default -std=c++20 -mmmx -msse -msse2 -msse3 -mssse3 '
            '-msse4 -msse4a -msse4.1 -msse4.2 -mavx -mavx2" '
            f'-DL2_CACHE_SIZE={l2_cache_size} ..'
        )
    elif system_name == "AVX-512":
        cmake_command = (
            'cmake -DCMAKE_C_COMPILER=/usr/bin/clang '
            '-DCMAKE_CXX_COMPILER=/usr/bin/clang++ '
            f'-GNinja -DCMAKE_BUILD_TYPE={build_mode} '
            '-DCMAKE_CXX_FLAGS="-Wno-switch-default -Wno-error=switch-default -std=c++20 -mmmx -msse -msse2 -msse3 -mssse3 '
            '-msse4 -msse4a -msse4.1 -msse4.2 -mavx -mavx2 '
            '-mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq '
            '-mavx512vnni" '
            f'-DL2_CACHE_SIZE={l2_cache_size} ..'
        )
    elif system_name == "Gracehopper":
        cmake_command = (
            'cmake -DCMAKE_C_COMPILER=/usr/bin/clang '
            '-DCMAKE_CXX_COMPILER=/usr/bin/clang++ '
            f'-GNinja -DCMAKE_BUILD_TYPE={build_mode} '
            '-DCMAKE_CXX_FLAGS="-Wno-switch-default -Wno-error=switch-default -std=c++20 -march=armv8.2-a+sve2" '
            f'-DL2_CACHE_SIZE={l2_cache_size} ..'

        )
    elif system_name == "Power10":
        cmake_command = (
            'cmake -DCMAKE_C_COMPILER=/usr/bin/clang '
            '-DCMAKE_CXX_COMPILER=/usr/bin/clang++ '
            f'-GNinja -DCMAKE_BUILD_TYPE={build_mode} '
            '-DCMAKE_CXX_FLAGS="-Wno-switch-default -Wno-error=switch-default -std=c++20 -mcpu=power10 -mvsx" '
            f'-DL2_CACHE_SIZE={l2_cache_size} ..'
        )
    else:
        raise ValueError(f"Unknown system name: {system_name}")
    
    # Run the cmake command as a single string
    process = subprocess.Popen(cmake_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Read output line by line in real time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
         break
        if output:
          print(output, end='')  # Print output as it comes in

    # Wait for the process to finish
    process.wait()

    
    process = subprocess.Popen('ninja hyriseBenchmarkSIMD', stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Read output line by line in real time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
         break
        if output:
          print(output, end='')  # Print output as it comes in

    # Wait for the process to finish
    process.wait()
  

def get_cpu_model():
    try:
        # Run the lscpu command
        result = subprocess.run(['lscpu'], capture_output=True, text=True, check=True)
        
        # Split the output into lines
        lines = result.stdout.splitlines()
        
        # Find the line containing the model name and extract it
        for line in lines:
            if "Model name" in line:
                # Split by ':' and strip whitespace
                model_name = line.split(':')[1].strip()
                return model_name
    except subprocess.CalledProcessError as e:
        print(f"Error while running lscpu: {e}")
        return None

def plot(l2_cache_size):
    print("Generating plot.")
    df = pd.read_csv('result.csv', header=None, 
                     names=['scale', 'time_sort', 'time_pdqsort', 'time_simd_sort', 'simd_speedup_sort', 'simd_speedup_pdqsort'])
    
    # Calculate num_values and throughputs
    num_values = 2**20 * df['scale']
    throughput_sort = num_values / (df['time_sort'] / 1000) / 1_000_000  # Throughput for std::sort (M tuples/s)
    throughput_pdqsort = num_values / (df['time_pdqsort'] / 1000) / 1_000_000  # Throughput for pdqsort (M tuples/s)
    throughput_simd_sort = num_values / (df['time_simd_sort'] / 1000) / 1_000_000  # Throughput for SIMD sort (M tuples/s)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Create an evenly spaced x-axis based on the number of scale entries
    x_positions = range(len(df['scale']))  # Evenly spaced x positions

    # Plot throughput for std::sort and pdqsort on the left y-axis
    ax1.plot(x_positions, throughput_sort, color='black', marker='o', label='std::sort')
    ax1.plot(x_positions, throughput_pdqsort, color='blue', marker='s', label='pdqsort')
    ax1.plot(x_positions, throughput_simd_sort, color='black', marker='x', label='simd_sort')

    ax1.set_xticks(x_positions)  # Set x-ticks to evenly spaced positions
    ax1.set_xticklabels(df['scale'].astype(str))  # Set x-tick labels to actual scale values
    ax1.set_xlabel('Number of tuples (in 2^20)')
    ax1.set_ylabel('Sort-throughput (M tuples/s)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    title = "[single-threaded, naive] " + get_cpu_model() + ", L2 (" + str(l2_cache_size) + "KiB)" 
    ax1.set_title(title)
    ax1.grid()

    # Create a second y-axis for speedup
    ax2 = ax1.twinx()
    
    # Set width of bars to slightly separate them
    bar_width = 0.4
    ax2.bar([x - bar_width / 2 for x in x_positions], df['simd_speedup_sort'], 
            color='gray', alpha=0.6, label='SIMD Speedup (std::sort)', width=bar_width)
    ax2.bar([x + bar_width / 2 for x in x_positions], df['simd_speedup_pdqsort'], 
            color='lightblue', alpha=0.6, label='SIMD Speedup (pdqsort)', width=bar_width)
    
    ax2.set_ylabel('Speedup', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(bottom=1)  # Set the lower limit to 1
    ax2.set_ylim(1, 2 * max(df['simd_speedup_sort'].max(), df['simd_speedup_pdqsort'].max()))

    # Show legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot.png')

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Create a build directory and run CMake for the specified system.")
    parser.add_argument("directory", type=str, help="The name of the directory to create.")
    parser.add_argument("system_name", type=str, choices=["AVX2", "AVX-512", "Gracehopper", "Power10"], help="The system architecture.")
    parser.add_argument('-c', '--cpr', type=int, default=4, help='Element count per SIMD register (default: 4)')
    parser.add_argument('-t', '--dt', type=str, default='double', help='Element data type (default: double)')
    parser.add_argument('-w', '--warumup', type=int, default=1, help='Number of warmup runs (default: 1)')
    parser.add_argument('-r', '--runs', type=int, default=5, help='Number of runs (default: 5)')
    parser.add_argument('--l2_cache_size', type=int, default=256, help='L2 cache size in KB (default: 256)')      
    parser.add_argument('--build_mode', type=str, default='Release', help='Build mode (default: Release)')
    args = parser.parse_args()

    print("Chosen configuration:")
    print(f"Directory to create: {args.directory}")
    print(f"System architecture: {args.system_name}")
    print(f"Element count per SIMD register: {args.cpr}")
    print(f"Element data type: {args.dt}")
    print(f"Number of warmup runs: {args.warumup}")
    print(f"Number of runs: {args.runs}")
    print(f"L2 Cache Size: {args.l2_cache_size} KiB")
    print(f"Build mode: {args.build_mode}")

     # Create the directory if it doesn't exist
    os.makedirs(args.directory, exist_ok=True)
    
    # Move into the directory
    os.chdir(args.directory)
    
    # Export configuration to config.txt
    config_path = os.path.join('config.txt')
    with open(config_path, 'w') as config_file:
        config_file.write("Chosen configuration:\n")
        config_file.write(f"Directory to create: {args.directory}\n")
        config_file.write(f"System: {args.system_name}\n")
        config_file.write(f"Element count per SIMD register: {args.cpr}\n")
        config_file.write(f"Element data type: {args.dt}\n")
        config_file.write(f"Number of warmup runs: {args.warumup}\n")
        config_file.write(f"Number of runs: {args.runs}\n")
        config_file.write(f"L2 Cache Size: {args.l2_cache_size} KiB\n")
        config_file.write(f"Build mode: {args.build_mode}")

    print(f"Configuration exported to {config_path}\n")

    # Call the create_and_build function with the provided arguments
    create_and_build(args.directory, args.system_name, 1024 * args.l2_cache_size, args.build_mode)

    print("Start benchmark.")

    command = [
        './benchmark',  # Replace with the path to your benchmark executable if necessary
        '-c', str(args.cpr),
        '-t', args.dt,
        '-w', str(args.warumup),
        '-r', str(args.runs),
        '-o', 'result.csv'
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Read output line by line in real time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
         break
        if output:
          print(output, end='')  # Print output as it comes in

    # Wait for the process to finish
    process.wait()

    # Check for errors
    stderr_output = process.stderr.read()
    if stderr_output:
       print(stderr_output, end='')
    plot(args.l2_cache_size)
    
  

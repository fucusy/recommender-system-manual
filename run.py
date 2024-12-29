import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_command(command):
    """
    Run a shell command and return the output or error.

    :param command: Command to run as a string or list of arguments.
    :return: Tuple containing the command and its result (stdout or stderr).
    """
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return e.stderr

def run_single_command_in_parallel(command, repetitions, max_workers=4):
    """
    Run a single shell command multiple times in parallel.

    :param command: The shell command to execute.
    :param repetitions: Number of times to execute the command.
    :param max_workers: Maximum number of threads to use.
    :return: List of results from each execution.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(run_command, command): i for i in range(repetitions)}
        for future in as_completed(future_to_index):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                results.append(f"Error: {exc}")
    return results

if __name__ == "__main__":
    # List of commands to run
    # generate a filename with current date,
    command = f"""python bandit_vs_learning_to_rank/weighted_agents.py"""
    # Run commands in parallel
    cpu_count = os.cpu_count()
    max_workers = cpu_count
    results = run_single_command_in_parallel(command, repetitions=10, max_workers=max_workers)

    # Print results
    for output in results:
        print(f"Command: {command}\nOutput: {output}\n")
    import datetime
    # writing the results to a file with a filename that is the current date and time
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/output_{date}.txt"

    with open(filename, "w") as f:
        for output in results:
            f.write(output)


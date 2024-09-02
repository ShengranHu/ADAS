import concurrent.futures
import random
import string
import numpy as np
import logging
import sys
import time

TASK_OVERVIEW = """You will be given some number of paired example inputs and outputs grids. The outputs were produced by applying a transformation rule to the input grids. In addition to the paired example inputs and outputs, there is also one test input without a known output.
The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). Each number corresponds to a color. 0 is black.
Your task is to determine the transformation rule from examples and find out the answer, involving determining the size of the output grid for the test and correctly filling each cell of the grid with the appropriate color or number.

The transformation only needs to be unambiguous and applicable to the example inputs and the test input. It doesn't need to work for all possible inputs. Observe the examples carefully, imagine the grid visually, and try to find the pattern.
"""

def visualize_image_using_emoji(*images, titles=None):
    emoji_map = ['⬛️', '🟦', '🟥', '🟩', '🟨', '⬜️', '🟪', '🟧', '🟪', '🟫']
    

    images = [np.array(image) if isinstance(image, list) else image for image in images]
    images = [np.argmax(image, axis=0) if len(image.shape) > 2 else image for image in images]
    
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    
    # Pad all images to the maximum size
    padded_images = []
    for image in images:
        pad_height = max_height - image.shape[0]
        pad_width = max_width - image.shape[1]
        padded_image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
        padded_images.append(padded_image)
    
    result = []
    if titles:
        column_width = max_width * 2 + 2  # Each emoji is 2 characters wide
        title_line = "".join(title.ljust(column_width) for title in titles)
        result.append(title_line)
    
    for h in range(max_height):
        line = []
        for image in padded_images:
            row = "".join(emoji_map[pixel] for pixel in image[h])
            line.append(row)
        result.append("  ".join(line))  # Two spaces between grids
    
    return "\n".join(result)

visualization_logger = None

def setup_logging(args, log_file, visualization_log_file):
    global visualization_logger
    
    # Configure the root logger
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file, encoding='utf-8'),
                        ])

    # Create a separate logger for visualizations
    visualization_logger = logging.getLogger('visualization')
    visualization_logger.setLevel(logging.INFO)
    viz_handler = logging.FileHandler(visualization_log_file, encoding='utf-8')
    viz_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    visualization_logger.addHandler(viz_handler)

    # Explicitly set the encoding for sys.stdout (for Windows compatibility)
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    logging.info("Logging setup complete. Unicode characters should now be handled correctly.")
    logging.info(f"Max API calls set to: {args.max_api_calls}")

    # Return the visualization_logger
    return visualization_logger

def reset_api_call_count_if_needed(api_tracker, current_time):
    if current_time - api_tracker['last_reset_time'] >= 60:
        api_tracker['api_call_count'] = 0
        api_tracker['last_reset_time'] = current_time

def check_max_api_calls(api_tracker, max_api_calls):
    if max_api_calls is not None and api_tracker['total_api_call_count'] >= max_api_calls:
        logging.info(f"Reached maximum API calls limit of {max_api_calls}. Stopping execution.")
        print(f"\nReached maximum API calls limit of {max_api_calls}. Stopping execution.")
        sys.exit(0)

def api_call_with_rate_limit(api_tracker, max_api_calls):
    current_time = time.time()
    reset_api_call_count_if_needed(api_tracker, current_time)
    check_max_api_calls(api_tracker, max_api_calls)
    
    if api_tracker['api_call_count'] >= 14:
        sleep_time = 60 - (current_time - api_tracker['last_reset_time'])
        if sleep_time > 0:
            logging.info(f"API call limit reached. Pausing for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
        reset_api_call_count_if_needed(api_tracker, time.time())
    
    api_tracker['api_call_count'] += 1
    api_tracker['total_api_call_count'] += 1

    return api_tracker

def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id


def file_to_string(filepath):
    with open(filepath, 'r') as f:
        data = f.read().strip()
    return data


def list_to_string(list_2d):
    sublists_as_strings = [f"[{','.join(map(str, sublist))}]" for sublist in list_2d]
    return f"[{','.join(sublists_as_strings)}]"


def format_arc_data(arc_data, direct=False):
    task_str = TASK_OVERVIEW

    task_demo_str = ''
    # Get task demo string
    task_demo_str += '## Examples:\n\n'
    for i, demo in enumerate(arc_data['train']):
        task_demo_str += f'### Example {i}:\n'
        task_demo_str += f'input = {list_to_string(demo["input"])}\n'
        task_demo_str += f'output = {list_to_string(demo["output"])}\n\n'

    # Get task test string
    task_test_str = ''
    for testcase in arc_data['test']:
        task_test_str += '## Test Problem:\n'
        task_test_str += f'Given input:\n {list_to_string(testcase["input"])}\n\n'
        task_test_str += f'Analyze the transformation rules based on the provided Examples and determine what the output should be for the Test Problem.'

    task_str += task_demo_str + task_test_str

    return task_str, arc_data['train'], arc_data['test'][0]['input']


def get_percentage_match(arr1, arr2):
    # arr1 is solution
    if not arr2:
        return 0
    score = 0
    for i, xs in enumerate(arr1):
        try:
            for j, x in enumerate(xs):
                try:
                    if len(arr2) > i and len(arr2[i]) > j and arr2[i][j] == x:
                        score += 1
                except:
                    pass
        except:
            pass
    score = score / (len(arr1) * len(arr1[0]))
    return score


def eval_algo(solve_fn, arc_data, soft_eval=False):
    # Calculate percentage of test cases done correctly
    testcases = arc_data['test']
    scores = []
    for testcase in testcases:
        input = testcase['input']
        output = testcase['output']
        gen_output = None
        # Run solve_fn with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(solve_fn, input)
                try:
                    gen_output = future.result(timeout=30)
                except concurrent.futures.TimeoutError:
                    future.cancel()
            except:  # if the function does not work
                continue
        # Check if correct output
        if soft_eval:
            score = get_percentage_match(output, gen_output)
        else:
            score = 1 if output == gen_output else 0
        scores.append(score)
    return np.mean(scores)


def eval_solution(output, arc_data, soft_eval=False):
    if not output:
        return 0

    solution = arc_data['test'][0]['output']
    if soft_eval:
        score = get_percentage_match(solution, output)
    else:
        score = 1 if output == solution else 0
    return score


def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
    """
    Calculate the bootstrap confidence interval for the mean of 1D accuracy data.
    Also returns the median of the bootstrap means.
    
    Args:
    - data (list or array of float): 1D list or array of data points.
    - num_bootstrap_samples (int): Number of bootstrap samples.
    - confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).
    
    Returns:
    - str: Formatted string with 95% confidence interval and median as percentages with one decimal place.
    """
    # Convert data to a numpy array for easier manipulation
    data = np.array(data)

    # List to store the means of bootstrap samples
    bootstrap_means = []

    # Generate bootstrap samples and compute the mean for each sample
    for _ in range(num_bootstrap_samples):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Compute the mean of the bootstrap sample
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    # Convert bootstrap_means to a numpy array for percentile calculation
    bootstrap_means = np.array(bootstrap_means)

    # Compute the lower and upper percentiles for the confidence interval
    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

    # Compute the median of the bootstrap means
    median = np.median(bootstrap_means)

    # Convert to percentages and format to one decimal place
    ci_lower_percent = ci_lower * 100
    ci_upper_percent = ci_upper * 100
    median_percent = median * 100

    # Return the formatted string with confidence interval and median
    return f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%"

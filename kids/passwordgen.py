#!/usr/bin/env python3

import argparse
import random
import string
import time
import itertools
import sys
from multiprocessing import Pool, cpu_count

def get_charset(level):
    if level == 1:
        return string.digits
    elif level == 2:
        return string.ascii_letters + string.digits
    elif level == 3:
        return string.ascii_letters + string.digits + string.punctuation
    else:
        raise ValueError("Invalid complexity level.")

# For random password generation
def generate_random(args):
    level, length = args
    charset = get_charset(level)
    return ''.join(random.choices(charset, k=length))

# For full enumeration (chunked)
def enumerate_chunk(start_end_charset_length):
    start, end, charset, length = start_end_charset_length
    results = []
    for i, combo in enumerate(itertools.islice(itertools.product(charset, repeat=length), start, end)):
        results.append(''.join(combo))
    return results

def main():
    parser = argparse.ArgumentParser(description="Multi-core Password Generator")
    parser = argparse.ArgumentParser(description="Only allowed for educational purposes! SSLab@Lewis Univ.")
    parser.add_argument('-D', action='store_true', help='Display output')
    parser.add_argument('-C', type=int, choices=[1, 2, 3], required=True, help='Complexity level: 1=numeric, 2=alphanumeric, 3=alphanumeric+special')
    parser.add_argument('-N', type=int, help='Password length (default: 4 for -C 1, 8 for others)')
    parser.add_argument('-G', type=int, default=10, help='Number of passwords to generate (0 = all combinations)')
    args = parser.parse_args()

    num_cores = cpu_count()
    level = args.C
    length = args.N if args.N else (4 if level == 1 else 8)
    charset = get_charset(level)
    start_time = time.time()

    # Case 1: Full combination generation
    if args.G == 0:
        total = len(charset) ** length

        if total > 1e7:
            print(f"WARNING: This will generate {total:,} combinations using all {num_cores} cores.")
            confirm = input("Continue? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("Aborted.")
                sys.exit(0)

        chunk_size = total // num_cores
        ranges = [(i * chunk_size, min((i + 1) * chunk_size, total), charset, length) for i in range(num_cores)]

        with Pool(processes=num_cores) as pool:
            results = pool.map(enumerate_chunk, ranges)

        if args.D:
            for chunk in results:
                print(' '.join(chunk))
        else:
            print(f"Generated {total:,} passwords in {time.time() - start_time:.2f} seconds using {num_cores} cores.")

    # Case 2: Random password generation
    else:
        task_args = [(level, length)] * args.G
        with Pool(processes=num_cores) as pool:
            results = pool.map(generate_random, task_args)

        if args.D:
            print(' '.join(results))
        else:
            print(f"Generated {args.G:,} passwords in {time.time() - start_time:.2f} seconds using {num_cores} cores.")

if __name__ == '__main__':
    main()

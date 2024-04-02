#!/usr/bin/env python3

"""
    Find the Mean, Median and Std Deviation of the characters in each webpage(document)
    of the website. Can be useful for creating chunks and identifying chunk overlap
"""

import os
import statistics
import matplotlib.pyplot as plt
import numpy as np

def character_count_stats(folder_path):
    character_counts = []

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            # Open the file and calculate character count
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                character_count = len(content)

                # Add character count to the list
                character_counts.append(character_count)

    # Calculate mean, median, and standard deviation
    if character_counts:
        mean = statistics.mean(character_counts)
        median = statistics.median(character_counts)
        std_dev = statistics.stdev(character_counts)
        minimum = min(character_counts)
        maximum = max(character_counts)
        length = len(character_counts)

        print(f"Mean Character Count: {mean}")
        print(f"Median Character Count: {median}")
        print(f"Standard Deviation: {std_dev}")
        print(f"Min Character count: {minimum}")
        print(f"Max Character count: {maximum}")
        print(f"Number of Documents: {length}")

        bins = [0, 100, 500, 750, 1000, 1500, 2000, np.inf]
        plt.hist(character_counts, bins=bins, color='skyblue', edgecolor='black')
        plt.show()
    else:
        print("No .txt files found in the folder.")

# Specify the folder containing the text files
folder_path = '../data/muenchen_en/'

# Call the function to find stats
character_count_stats(folder_path)

"""
file: HW_06_Classifier_Kallurwar.py
description:
language: python3
author: Anurag Kallurwar, ak6491@rit.edu
"""

import warnings
import sys
import os
import math
import numpy as np
import pandas as pd


# CONSTANTS
PREDICTION = 'ClassID'
PREDICTION2 = 'ClassName'
CLASSIFICATION_OUTPUT_FILE = "HW_06_Kallurwar_MyClassifications.csv"


def clean_data(df: pd.DataFrame):
	"""
	Cleaning the dataframe
	:param df: input dataframe
	:return: cleaned dataframe
	"""
	return df.dropna()


def read_file(file_name: str):
	"""
	Read the CSV file and return dataframe
	:param thread_index: Index of thread
	:param file_paths: filename
	:return: dataframe
	"""
	print("Reading file: " + file_name)
	# Skipping first line containing "HEADER" string
	dataframe = pd.read_csv(file_name, low_memory=False)
	dataframe = clean_data(dataframe)
	return dataframe


def classify(input_df: pd.DataFrame):
	"""
	This is the resultant classifer of the decision tree which classifies the
	given input
	:param input_df:
	:return:
	"""
	attributes = input_df.columns.tolist()
	lines = [','.join(attributes + [PREDICTION2, PREDICTION]) + "\n"]
	for index, row in input_df.iterrows():
		row_dictionary = dict()
		for index in range(len(attributes)):
			row_dictionary[attributes[index]] = row[index]
		#START
		if row_dictionary['BangLn'] < 5.6000000000000005:
			if row_dictionary['HairLn'] < 12.375:
				if row_dictionary['ApeFactor'] < 6.375:
					if row_dictionary['TailLn'] < 2.625:
						class_label = 1
					else:
						if row_dictionary['Age'] < 18.2:
							class_label = -1
						else:
							if row_dictionary['Shagginess'] < 6.099999999999999:
								if row_dictionary['Ht'] < 119.29999999999998:
									class_label = -1
								else:
									class_label = -1
							else:
								if row_dictionary['EarLobes'] < 0.9000000000000002:
									class_label = -1
								else:
									class_label = -1
				else:
					class_label = 1
			else:
				class_label = 1
		else:
			if row_dictionary['ApeFactor'] < 4.5:
				class_label = -1
			else:
				if row_dictionary['Age'] < 63.800000000000026:
					if row_dictionary['HairLn'] < 9.500000000000002:
						if row_dictionary['Shagginess'] < -2.1:
							class_label = 1
						else:
							if row_dictionary['TailLn'] < 2.475:
								class_label = 1
							else:
								if row_dictionary['Ht'] < 175.7000000000001:
									class_label = 1
								else:
									class_label = -1
					else:
						class_label = 1
				else:
					class_label = -1
		#END
		class_value = "Assam"
		if class_label == 1:
			class_value = "Bhuttan"
		print(class_label, class_value)
		lines += [",".join(map(str, row)) + "," + str(class_value) + "," + str(class_label) + "\n"]
	return lines


def write_output(ouput: str):
	"""
	Creates a CSV output file
	:param decision_tree: Decision Tree
	:return: None
	"""
	with open(CLASSIFICATION_OUTPUT_FILE, 'w') as file:
		file.writelines(ouput)


def process(abonimable_df: pd.DataFrame):
	"""
	This function compute features and predictions the classID for the input data
	:param abonimable_df: Input data
	:return: None
	"""
	# Part 1.
	# Quantize Data
	print("\n\n===== Quantizing Data =====")
	# Quantizing Age to nearest 1 years
	abonimable_df['Age'] = abonimable_df['Age'].round(0)
	# Quantizing Ht to nearest 1 cm
	abonimable_df['Ht'] = abonimable_df['Ht'].round(0)
	# Quantizing anything else to nearest 1/2 value
	columns_to_be_quantized = ['TailLn', 'BangLn', 'HairLn', 'Reach']
	# For every value in the above columns is being rounded
	abonimable_df = abonimable_df.apply(lambda value: (round(value * 2) / 2)
	if value.name in columns_to_be_quantized else value)

	# Part 2
	# Feature Generation
	abonimable_df['Shagginess'] = abonimable_df['HairLn'] - abonimable_df['BangLn']
	abonimable_df['ApeFactor'] = abonimable_df['Reach'] - abonimable_df['Ht']
	print(abonimable_df)

	# Part 3.
	# Classification for input data
	print("\n\n===== CLASSIFICATION =====")
	lines = classify(abonimable_df)
	print("\nWriting output to " + CLASSIFICATION_OUTPUT_FILE)
	write_output(lines)


def main():
	"""
	The main function
	:return: None
	"""
	warnings.simplefilter(action='ignore', category=FutureWarning)
	if len(sys.argv) < 2:
		print("Missing Argument!")
		print("Usage: HW_06_Classifier_Kallurwar.py <filename.csv>")
		return
	file_name = sys.argv[1].strip()
	if not os.path.isfile(os.getcwd() + "\\" + file_name):
		print("Please put " + file_name + " in the execution folder!")
		return
	abonimable_df = read_file(file_name)
	process(abonimable_df)


if __name__ == '__main__':
	main()  # Calling Main Function

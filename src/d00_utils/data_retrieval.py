import boto3
import cv2
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


from io import BytesIO
import re
import shutil
import datetime

def retrieve_videos_from_s3(bKeep_data=True, from_date=None, to_date=None):
	"""Retrieve jamcam videos from the s3 bucket based on the dates specified.
	Downloads to a local directory and then loads them into numpy arrays.

		Args:
			bKeep_data: boolean for keeping the downloaded data in the local folder
			from_date: start date (inclusive, YYYY-MM-DD) for retrieving videos, if None then will retrieve from 2019-06-01 onwards
			to_date: end date (inclusive,, YYYY-MM-DD) for retrieving vidoes, if None then will retrieve up to current day
		Returns:
			list of numpy arrays containing all the jamcam videos between the selected dates
		Raises:

	"""

	# Deal with the cases when a date isn't specified
	if (from_date == None):
		from_date = '2019-06-01'
	if (to_date == None):
		to_date = str(datetime.datetime.now())[:10]

	# Set local directory for downloading data, will overwrite whatever is currently there
	local_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
							 '..', '..', 'data/01_raw/jamcams/')
	if (os.path.isdir(local_dir)):
		shutil.rmtree(local_dir)
	os.mkdir(local_dir)

	# Set up boto3 session
	s3_session = boto3.Session(profile_name='dssg')
	bucket_name = 'air-pollution-uk'
	s3_resource = s3_session.resource('s3')
	my_bucket = s3_resource.Bucket(bucket_name)

	# Get list of files in s3 based on dates provided
	selected_files = []
	objects = my_bucket.objects.filter(Prefix="raw/video_data_new/")
	files = [obj.key for obj in objects]
	for file in files:
		date = re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2})", file).group()
		if (date >= from_date and date <= to_date):
			selected_files.append(file)

	# Download the selected files
	data = []
	for file in selected_files:
		my_bucket.download_file(file, local_dir + file.split('/')[-1])

	# Load files into a list of numpy arrays using opencv
	for file in os.listdir(local_dir):
		cap = cv2.VideoCapture(local_dir + file)

		frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
		fc = 0
		ret = True
		while (fc < frameCount and ret):
			ret, buf[fc] = cap.read()
			fc += 1
		cap.release()
		data.append(buf)

	# Delete local data unless specified
	if(not bKeep_data):
		shutil.rmtree(local_dir)

	return data


def retrieve_tims_from_s3(from_date='2019-06-01', to_date=str(datetime.datetime.now())[:10]):
	'''
	Retrieves TIMS CSV data from the S3 bucket between the specified range of dates (inclusive)

		Args:
			from_date: format YYYY-MM-DD
			to_date: format YYYY-MM-DD, default is today

		Returns: a single pandas dataframe, created by concatenating all the separate TIMS CSVs together

		Raises:

	'''
	#typecast from and to dates 
	from_YYYY, from_mm, from_dd = from_date.split("-")
	from_date = datetime.date(year = int(from_YYYY), month = int(from_mm), day = int(from_dd))
	to_YYYY, to_mm, to_dd = to_date.split("-")
	to_date = datetime.date(year = int(to_YYYY), month = int(to_mm), day = int(to_dd))

	#names and paths 
	bucket_name = 'air-pollution-uk'
	tims_path = 'raw/tims_data/' 

	#setup s3 retrieval with boto3
	s3_session = boto3.Session(profile_name='dssg')
	s3_client = s3_session.client('s3')
	s3_resource = s3_session.resource('s3')

	my_bucket =  s3_resource.Bucket(bucket_name)

	#get names of all tims files in s3 based on provided dates
	objects = my_bucket.objects.filter(Prefix=tims_path)

	# Get list of files in s3 based on dates provided
	selected_files = []
	filepaths = [obj.key for obj in objects]
	for fp in filepaths:
		#get dates for each fp
		filename = fp.split("/")[-1]
		date_str = re.search("(detdata)([0-9]*)", filename).group(2)
		#TIMS data dates are in format ddmmYYYY
		dd, mm, YYYY = "", "", ""
		dd, mm,YYYY = int(dd.join(date_str[0:2])), int(mm.join(date_str[2:4])), int(YYYY.join(date_str[4:]))
		filedate = datetime.date(year=YYYY, month=mm, day=dd )

		if (filedate >= from_date and filedate <= to_date):
			selected_files.append(fp)

	#loop through selected files, retrieve csv obj from S3 bucket using the selected filepaths
	df_list = []
	for fp in selected_files: 
		csv_obj = s3_client.get_object(Bucket=bucket_name, Key=fp)
		df_list.append(pd.read_csv(csv_obj['Body'])) 

	return pd.concat(df_list)


def retrieve_cam_details_from_psql():
	'''
		Args:
		Returns: 
		Raises:
	'''

	return


def describe_s3_videos():
	"""
	Plot the number of videos in the s3 bucket for each date.
	Plot is saved locally under plots/01_exploratory.

		Args:

		Returns:

		Raises:

	"""

	# Set up boto3 session
	s3_session = boto3.Session(profile_name='dssg')
	bucket_name = 'air-pollution-uk'
	s3_resource = s3_session.resource('s3')
	my_bucket = s3_resource.Bucket(bucket_name)

	# Get list of all dates in the s3 bucket
	objects = my_bucket.objects.filter(Prefix="raw/video_data_new/")
	files = [obj.key for obj in objects]
	dates = []
	for file in files:
		res = re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2})", file)
		dates.append(res.group())

	# Plot a bar chart of the dates in the s3 bucket
	unique_dates, counts = np.unique(dates, return_counts=True)
	plt.figure()
	plt.bar(np.arange(unique_dates.shape[0]), counts)
	plt.xticks(np.arange(unique_dates.shape[0]), unique_dates, rotation='vertical')
	plt.xlabel('Date')
	plt.ylabel('Number of Videos')
	plt.tight_layout()
	plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)),
									  '..', '..', 'plots/01_exploratory/s3_video_description.pdf'))
	plt.close()

	return


def describe_s3_tims(): 
	'''
	Plot the number of TIMS CSVs in the S3 bucket for each date.
	Plot is saved locally under plots/01_exploratory.

		Args: None
		Returns: None
		Raises: None

	'''

	#names and paths 
	bucket_name = 'air-pollution-uk'
	tims_path = 'raw/tims_data/' 

	#setup s3 retrieval with boto3
	s3_session = boto3.Session(profile_name='dssg')
	s3_resource = s3_session.resource('s3')

	my_bucket =  s3_resource.Bucket(bucket_name)

	#get names of all tims files in s3 based on provided dates
	objects = my_bucket.objects.filter(Prefix=tims_path)

	filepaths = [obj.key for obj in objects]
	datecounter_dict = defaultdict(int)


	for fp in filepaths:
		#get dates for each fp
		filename = fp.split("/")[-1]
		if filename.startswith("detdata"): 
			date_str = re.search("(detdata)([0-9]*)", filename).group(2)
			datecounter_dict[date_str] +=1
		
	dates = []
	counts = []
	for key in datecounter_dict: 
		#recall dmy
		dd, mm,YYYY = key[0:2], key[2:4], key[4:]
		formatted_date = YYYY + "-" + mm + "-" + dd
		dates.append(formatted_date)
		counts.append(datecounter_dict[key])

	#plot counts of csvs with each date and save in directory
	plt.figure()
	plt.bar(np.arange(len(dates)), counts)
	plt.bar(dates, counts)

	plt.xticks(np.arange(len(dates)), dates, rotation='vertical')
	plt.xlabel('Date (YYYY-mm-dd)')
	plt.ylabel('Number of CSVs')
	plt.tight_layout()
	plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)),
									  '..', '..', 'plots/01_exploratory/s3_tims_description.pdf'))
	plt.close()

	return 
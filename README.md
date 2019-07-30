# Quantifying Traffic Dynamics from Traffic Videos 

Better insight extraction from traffic data will aid the effort to both understand and reduce air pollution. Currently, traffic statistics are obtained through high-cost manual labor (i.e. individuals counting vehicles by the road) and extrapolated to annual averages. Furthermore, they are not detailed enough to evaluate traffic/air pollution initiatives. The purpose of the project is to create an open-source library to automate the collection of live traffic video data and traffic induction data, and extract descriptive statistics. With this library, project partners would be able to obtain real-time (every 15 minutes) traffic statistics which are localized to the level of individual streets, easily and quickly. 

## Table of Contents

1. [Introduction](#introduction)
2. [Code Overview](#code-overview)
3. [Repo Structure](#repo-structure)
4. [Contributors](#contributors)
5. [License](#license)

## Introduction

<p float="left">
  <img src ="readme_resources/images/dssg_imperial_logo.png" alt="alt text" width="500" height="175"  />
</p> 

### Data Science for Social Good at Imperial College London

Data Science for Social Good (DSSG) summer fellowship is a 3-months long educational program of Data Science for Social Good foundation and the University of Chicago, where it originally started. In 2019, the program is organized in collaboration with the Imperial College Business School London, and Warwick University and Alan Turing Institute. Two cohorts of about 20 students each are hosted in these institutions where they are working on projects with our partners. 
The mission of the summer fellowship program is to train aspiring data scientists to work on data mining, machine learning, big data, and data science projects with social impact, with a special focus on the ethical consequences of the tools and systems they build. Working closely with governments and nonprofits, fellows take on real-world problems in education, health, energy, public safety, transportation, economic development, international development, and more.


### Partners

- The **Transport and Environmental Laboratory** (Department of Civil and Environmental Engineering, Imperial College London) has a mission to advance the understanding of the impact of transport on the environment, using a range of measurement and modelling tools. 

- **Transport for London** (TFL) is an integrated transport authority under the Greater London Authority,  which runs the day-to-day operation of London’s public transport network and manages London’s main roads. TFL leads various air quality initiatives including the Ultra Low Emission Zone in Central London.

- **City of London Corporation** (CLC) is the governing body of the City of London (also known as the Square Mile), one of the 33 administrative areas into which London is divided. Improving air quality in the Square Mile is one of the major issues the CLC will be focusing on over the next few years. 

- The  **Alan Turing Institute** is the national institute for data science and artificial intelligence. Its mission is to advance world-class research in data science and artificial intelligence, train leaders of the future, and lead the public conversation. 


## Code Overview
#### Installation and setup

Provide setup instructions.
If there are any specific hardware requirements, please specify here.

#### Code Example

Show a new user how to use the package.

#### Software
This project requires Python 3.7 or later, with packages as specified in requirements.txt. If you have pip installed, packages can be installed by running `pip install -r requirements.txt`.

*CUDA for using GPUs:* We used version ________.

*PostgreSQL for storing non-video data for inputs and outputs to the model.* We used version ________. You should have the command line executable, psql.

*Computer Vision Annotation Tool (CVAT)*: We used version _________. Instructions on installing and running CVAT can be found on their website.

*Docker for running a docker container that contains CVAT:* We used version __________.

#### Other services 
We used AWS virtual machines and S3 buckets to store video data, frame level dataframes, video level dataframes, and results of evaluation. 

#### Configuration Files
System specifications such as model parameters and file paths are contained in `YAML` configuration files found in `air_pollution_estimation/conf/base/`. In `run_pipeline.py`, our system reads the conf files from the aforementioned directory into dictionaries. 

Stored in this directory are the following config files: 
* `app_parameters.yml` contains configuration options for the web app.
* `paths.yml` contains relevant file paths for input and output files, as well as paths to temp folders used for temporary downloading/uploading of data.
* `parameters.yml` defines various hyperparameters for detection and tracking. It also contains configuration options for reporting.

We recommend that credentials be stored in a git-ignored `YAML` file in `air_pollution_estimation/conf/local/credentials.yml`
* `credentials.yml` should contain credentials necessary for accessing the PostgreSQL database, Amazon AWS services.

## Repo Structure 
<p float="left">
  <img src ="readme_resources/images/s3_structure.png" alt="alt text" />
</p> 

#### Data loading pipeline
#### Video analysis pipeline
*Main tasks: get videos from S3, initialize model, run model to construct frame level dataframes (bounding box positions) and video level dataframes of traffic statistics (counts, starts/stops).*

######  Modelling overview: 

All models should inherit from the TrafficAnalyserInterface. The TrafficAnalyserInterface requires that any child classes implement  construct_frame_level_df() and construct_video_level_df() methods. 

- Version 1: YOLO object detection only  
- Version 2: YOLO objects detection with tracking 

#### Evaluation 

- ChunkEvaluator is implemented using SingleEvaluator
- Frame level evaluation — mean average precision
- Video level evaluation  —mean/std dev difference, mean squared error 

## Contributors

Jack Hensley, Oluwafunmilola Kesa, Sam Blakeman, Caroline Wang, Sam Short (Project Manager), Maren Eckhoff (Technical Mentor)

## License

Fill in later
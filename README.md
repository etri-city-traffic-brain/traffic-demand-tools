# traffic-demand-tools
## Task 1. TOD plan based on smart-intersection data
### Sample Code
python TOD.py --crsrd-id 1850041700 --input-dir ./data --output-dir ./result --max-tod 4

#### Arguments
* crsrd-id : smart-intersection ID
* input_dir : directory including input datasets
* output-dir : directory to save the results
* max-tod : maximum number of TOD groups (> 1)

### 1-1. Description
Using dataset from smart-intersection, the time table with TOD labels is estimated by K-means method
- Time units: 30 minutes
- Single intersection(crossroad)
- Note: Go-direction traffic includes right-turn traffic
        TOD considers day types - weekdays, saturday, and sunday

### 1-2. Requairements
#### Environment/Packages/Libraries
* Python 3
* pandas
* dplython
* scikit-learn

#### Input datasets
* ORT_CCTV_5MIN_LOG
* ORT_CCTV_MST

### 1-3. Results
#### TOD table
* For each day type, time (30 min-unit), TOD is labeled
* Example

#### Traffic Analysis
* Traffic characteristics according to each TOD period
- turning rate
- total traffic (veh/30min)
* Example

### 1-2. Issues
#### Missing values in input data (ORT_CCTV_5MIN_LOG)
* (1st) moving average
* (2nd) fill with 0 and drop na or inf values during normalization
* Still need to improve

## Task 2. RSE Analysis (for target area)
### Sample Code
python uniq_rse_analysis.py ./data/RSE_COL_20211124.xlsx 20211124

#### Arguments
* Target RSE Data File Name
* Target Date (on which the RSE Data is collected)

### Description
* Analyze the travel time between specific RSE spots, and draw/store the Boxplot Figure of the travel time according to RSE spots.
- Output Example


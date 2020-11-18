# traffic-demand-tools
## Task 1. TOD plan based on smart-intersection data
### Sample Code
python TOD.py --crsrd-id 1860001400 --input-dir ./data --output-dir ./result --max-tod 5
- Arguments
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
Environment/Packages/Libraries
* Python 3
* pandas
* dplython
* scikit-learn

Input datasets
* ORT_CCTV_5MIN_LOG
* ORT_CCTV_MST

### 1-3. Results
TOD table
* For each day type, time (30 min-unit), TOD is labeled
* Example

Traffic Analysis
* Traffic characteristics according to each TOD period
- turning rate
- total traffic (veh/30min)
* Example

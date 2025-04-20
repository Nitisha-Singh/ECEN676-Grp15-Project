# ECEN676-Project-Grp15: A Cost-Effective Entangling Prefetcher for Instructions *(EPI)* [paper](https://ieeexplore.ieee.org/document/9499798)
The Entangling Prefetcher is an advanced instruction prefetching mechanism designed to improve processor performance by reducing instruction cache (L1I) misses,
especially in workloads with large instruction footprints such as cloud and server applications. Unlike traditional prefetchers that use a fixed look-ahead distance
or simple sequential patterns, the Entangling Prefetcher dynamically learnsand pairs ("entangles") cache lines based on the observed latency of instruction cache misses.
## Setup Instructions
### 1. Clone Repository
```
git clone https://github.com/cpepis/ChampSim.git
git checkout wrong-path
git fetch
```
Follow the instructions mentioned in README file of cloned repository for downloading the dependencies.
### 2. Download Traces
Download traces from [Google Drive](https://drive.google.com/drive/folders/1XQQ2FJz97jm_Bweq2of_3gzHzk022EEd) and place them in folder named TRACES in the working directory.
### 3. Configuration
#### 1. Champsim config
Update the following parameters to configure Champsim by update/ creating config file (like champsim_config_epi.json) within config folder.
```
"executable_name": "champsim_epi"
    "L1I": {
        "sets": 64,
        "ways": 8,
        "rq_size": 64,
        "wq_size": 64,
        "pq_size": 32,
        "mshr_size": 8,
        "latency": 4,
        "max_tag_check": 2,
        "max_fill": 2,
        "prefetch_as_load": true,
        "virtual_prefetch": true,
        "prefetch_activate": "LOAD,PREFETCH",
        "prefetcher": "epi_instr"
    }
```
#### 2. EPI prefetcher config
Update the following in prefetcher/epi_instr/epi.cc to configure the prefetcher for 2K, 4K and 8K entries
```
#define L1I_ENTANGLED_TABLE_INDEX_BITS 8    //2k= 6 4k=7 and 8k=8
```
After creating and updating the configs build champsim using:
```
./config.sh <champsim_config_filename>.json
make
```
### 4. Job script creation and submission
Update the email address in the job scripts provided in the jobs folder.
Submit the jobs using:
```
sbatch <folder>/<script_name>.sh
```
### 5. Results 
#### 1. Create folder name results in the Champsim folder with the following directory structure to save the results:
```
results
|-nop
|-epi_2k
|-epi_4k
|-epi_8k
|-epi_16k
|-next
|-mana
|-djolt
|-fnlmma
```
### 2. For analysing the results place analyse.py in the folder containing the results folder and execute it using following:
```
Navigate to the folder containing analyse.py
python3 analyse.py
```
Successfull execution will generate *simulation_analysis_results.xlsx* file
### Group Members
1. Nitisha Singh
2. Rajat Sharma
3. Quy Van

# Deep Pedestrian Trajectory Prediction for Trffic Intersection (DeepPTP)
**Author: Zhiqiang Lv**

**Institute of Ubiquitous Networks and Urban Computing, Qingdao University, China**
 ## Data Set
 https://www.th-ab.de/ueber-uns/organisation/labor/kooperative-automatisierte-verkehrssysteme/trajectory-dataset/
 ## Requirements
  
  * Torch == 1.2.0
  * TorchVision==0.4.0
  * NumPy == 1.18.1
  * Python == 3.7
  
 ## install
 `pip install -r requirements.txt`
 
 ## Run
 `python3 main.py`
 ## Directory Structure
 ~~~~

├── model            
     ├── DeepPTP.py           // The model of DeepPTP
     ├── SpatialConv.py       // The Spatial Convolation Layer
     ├── TemporalConv.py      // The Temporal Convolation Layer
├── logs            
     ├── Run.log              // The log of train and evalution process
├── utils.py                 // Custom method
├── DataLoad.py              // Load data set
├── Train.py                 // Train process
├── Evalution.py             // Evalution process
├── Main.py                  // Main method
├── config.json              // Config information
├── requirements.txt         // Project interpreter
├── README.md                // Introduction
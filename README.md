# Eye movement event level matching with IoU
There are various different methods of event level matching for eye movements and this is a Python implementation of using Intersection over Union (IoU) as a threshold to decide how big of an overlap there needs to be between two events for them to be a hit. This can be used to calculate and event level F1-score for e.g. fixation, saccades, PSO and smooth pursuit. This event level matching is described in the paper *1D CNN with BLSTM for automated classification of fixations, saccades, and smooth pursuits* by Startsev et al. 2018 from the journal *Behavior Research Methods*

## Pseudo code
To be added later



## Installation

Clone the repositry and import eventLevelMatchingIoU.py in your project. Make sure to have the Python packages Numpy, Pandas and Copy installed. In case functionallity of the packages change the used versions of each package is listed here:


Numpy: 1.14.5


Pandas: 0.23.4

## Usage

```python
import eventLevelMatchingIoU

alg = [0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,1,1,1,1,0,0,0,2,1,1,1,1,1]
gt = [0,0,0,1,1,1,1,1,1,0,0,0,2,2,2,2,2,3,3,3,3,3,1,1,1,1,0,0,0,0,1,1,1,0,1]
container = eventLevelMatchingIoU(gt,alg)
```
By default a dict is returned where the keys are the labels of the arrays, e.g. 0,1,2,3. The output contains an event level F1-score and number of hits, false alarms type 1 and type 2 and misses. It also contains the arrays that show which events where what type. So the array [event_type_name+'_hit'] contains 1's for the events that were a hit and 0 for those that werent, while event_type_name+'_hit_counter' is the number of hits.

If you want the output to have different names you can do:

```python
container_2 = eventLevelMatchingIoU(gt,alg,label_names = ['a','b','c','d'])
```
If you are only interested in some of the event types and not all event types you can specifty this by

```python
container_1 = eventLevelMatchingIoU(gt,alg,label_names = None,event_types = [0,3])
```
If you want to specify both label_names and event_types make sure so specify the same amount, e.g.: 
label_names = ['name1','name3']
event_types = [1,3]

## Author
Shagen Djanian
2019-07-24

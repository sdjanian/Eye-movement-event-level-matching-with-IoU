"""
    This is an implementation of event level matching for eye movement classification using Intersecion over Union (IoU).
    The evaluation is described by Startsev, M., Agtzidis, I., and Dorr, M., 2018, in "1D CNN with BLSTM for
    automated classification of fixations,saccades, and smooth pursuits" but no implementation was available.
    
    Usage:
        alg = [0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,1,1,1,1,0,0,0,2,1,1,1,1,1]
        gt = [0,0,0,1,1,1,1,1,1,0,0,0,2,2,2,2,2,3,3,3,3,3,1,1,1,1,0,0,0,0,1,1,1,0,1]
        container = eventLevelMatchingIoU(gt,alg)
        
    Author:
        Shagen Djanian
    Date:
        2019-07-24
"""
import numpy as np
import pandas as pd
import copy


def event_level_f1_score(hits,false_alarm,miss):
  '''
      Calcuates the event level f1-score.
      
      Parameters:
      -----------
          hits : number of hits.
          false_alarm : the number of false alarms. Type 1 and type 2 false alarms should be grouped as one type of false alarm.
          miss: number of misses.
    
      Returns:
      --------          
          event_level_f1_score : The event level f1-score
    
      References:
      ----------
          Hooge, I. T., Niehorster, D. C., Nyström, M., Andersson,
          R., and Hessels, R. S. 2017, "Is human classification by experienced
          untrained observers a gold standard in fixation detection?",
          Handle Outliers", Behavior Research Methods.
  '''
  try:
      event_level_f1_score = (2*hits)/(2*hits+miss+false_alarm)
      return event_level_f1_score
  except ZeroDivisionError:
      return 0

def eventLevelMatchingIoU(gt,alg,IoU_threshold = 0.5,label_names = None,event_types = None):
  '''
      Matches events labels from an algorithm with events in the ground truth. 
      Intersection over union is as a threshold for what minimum of overlap is accepted for a hit.
      
      Parameters:
      -----------
            gt  : The labels of the ground truth on a sample level.
            alg : The labels of the algorithm which is being compared to the ground truth. Also sample level.
            IoU_threshold : The Intersecion over Union threshold which determines the acceptable overlap for a hit. Default is 0.5
            label_names : The name that the labels represent e.g.  0 is fixation, 1 is saccade. Used as ['name1','name2','name3'].
                          Default is None, so the actual labels are used as names.
            event_types : The types of events that you want to compare. If your labels are 0,1,2,3 and you only interested in event 1 and 3. Used [1,3]. 
                          Default is None. If both label_names and event_types are used they must be of same length. 
                          E.g:
                              label_names = ['name1','name3']
                              event_types = [1,3]
    
        Returns:
        --------
            output_2 : A dict comprised the dicts for each event type and the overall result. It contains the f1-score and number of hits, false alarms type 1 and type 2 and misses.
                       It also contains the arrays that show which events where what type. So the array [event_type_name+'_hit'] contains 1's for the events 
                       that were a hit and 0 for those that werent, while event_type_name+'_hit_counter' is the number of hits.
    
        References:
        ----------
            Startsev, M., Agtzidis, I., and Dorr, M., 2018, "1D CNN with BLSTM for
            automated classification of fixations,saccades, and smooth pursuits",
            Behavior Research Methods.
  '''
   
  # They need to be panda series because index manipulation is easier in pandas
  if type(gt) is not pd.Series():
    gt = pd.Series(gt)
  if type(alg) is not pd.Series():
    alg = pd.Series(alg)
  output = []
  output_2 = dict()
  if event_types is None:
      event_types = np.union1d(np.unique(gt),np.unique(alg))
  gt_event_number, gt_event_start,gt_event_end = getEventStartsAndEnds(gt)
  gt_event_number = pd.Series(gt_event_number)
  alg_event_number, alg_event_start,alg_event_end = getEventStartsAndEnds(alg)
  alg_event_number = pd.Series(alg_event_number)
  for i_label_name,event_type in enumerate(event_types,start=0):
    #print('label name: ',i_label_name,' event type: ',event_type )
    
    gt_current_event = copy.deepcopy(gt)
    gt_current_event[gt!=event_type] = 0
    gt_current_event[gt==event_type] = 1

    alg_current_event = copy.deepcopy(alg)
    alg_current_event[alg!=event_type] = 0
    alg_current_event[alg==event_type] = 1


    hit = np.zeros(len(gt))
    false_positive = np.zeros(len(gt))
    false_positive_type_2 = np.zeros(len(gt))
    miss = np.zeros(len(gt))
    IoU_array = np.zeros(len(gt))
    
    # Keep track of what events have already been labelled
    already_assigned_label_alg = np.zeros(len(gt)) # Need to be finished
    already_assigned_label_gt = np.zeros(len(gt)) # Need to be finished

    hit_counter = 0
    false_positive_counter = 0
    miss_counter = 0
    for event in np.unique(gt_event_number[gt_current_event==1]):
      gt_event = gt_current_event[gt_event_number==event]
      alg_events = alg_current_event[gt_event.index]
      number_of_same_samples_detected = gt_event == alg_events
      number_of_alg_events_occuring_during_gt_event = np.unique((alg_event_number.where(number_of_same_samples_detected).dropna()))
      #print(number_of_alg_events_occuring_during_gt_event.min())
      # Count the number of misses

      if number_of_alg_events_occuring_during_gt_event.size == 0:
        #print('label name: ',i_label_name,' event type: ',event_type,
        #      'numer overlapping events: ',number_of_alg_events_occuring_during_gt_event,' ',number_of_alg_events_occuring_during_gt_event.size,
        #      ' ',number_of_same_samples_detected)
        miss_counter = miss_counter + 1
        miss[gt_event.index] = 1
        already_assigned_label_gt[gt_event.index] = 1
        
      else:
        for alg_event_during_gt in number_of_alg_events_occuring_during_gt_event:
          gt_event_being_evaluated_index = gt_event_number[gt_event_number==event].index
          alg_event_being_evaluated_index = alg_event_number[alg_event_number==alg_event_during_gt].index
          if all(already_assigned_label_alg[alg_event_being_evaluated_index]==1):  
              # Skip events in the algorithm that have already been labelled, so they don't get labelled twice
              #print('skip',event_type,alg_event_during_gt)
              #print(alg_event_being_evaluated_index)
              None
          else:
              union = np.union1d(gt_event_being_evaluated_index,alg_event_being_evaluated_index)
              intersection = np.intersect1d(gt_event_being_evaluated_index,alg_event_being_evaluated_index)
              IoU = len(intersection)/len(union)
              if IoU > IoU_threshold:
                hit_counter = hit_counter +1
                hit[gt_event.index] = 1
                IoU_array[gt_event.index] = IoU
                already_assigned_label_gt[gt_event.index] = 1
                already_assigned_label_alg[alg_event_being_evaluated_index]=1
              else:
                false_positive_counter = false_positive_counter +1
                false_positive[alg_event_being_evaluated_index] = 1
                already_assigned_label_alg[alg_event_being_evaluated_index] = 1
              #already_assigned_label_alg[alg_event_being_evaluated_index]=1 # Keep track of algorithm event that have already been evaluated

        # Any ground truth events that were not a hit are set as misses
        if any(already_assigned_label_gt[gt_event.index] == 0):
            #print('label name: ',i_label_name,' event type: ',event_type,
            #  'numer unmatched events: ', already_assigned_label_gt)
            miss_counter = miss_counter + 1 # Lee's version
            miss[gt_event.index] = 1 # Lee's version 
            already_assigned_label_gt[gt_event.index] = 1
        
        # Get the unmatched event from the algorithm stream
        #print(np.unique(alg_current_event[alg_current_event==1]))
    # Count the number false alarms where an event occurs in the algorithm stream but not in the ground truth
    for alg_event_during_not_during_gt in np.unique(alg_event_number[alg_current_event==1]):
        if any(already_assigned_label_alg[alg_event_number[alg_event_number==alg_event_during_not_during_gt].index]==0):
            false_positive_counter = false_positive_counter +1
            false_positive_type_2[alg_event_number[alg_event_number==alg_event_during_not_during_gt].index] = 1  
            hit[alg_event_number[alg_event_number==alg_event_during_not_during_gt].index] = 0
            #already_assigned_label_alg[alg_event_number[alg_event_number==alg_event_during_not_during_gt].index] = 1
            #print("unmatched",i_label_name)
    '''
    for alg_event_during_not_during_gt in np.unique(alg_event_number[alg_current_event==1]):
      alg_event_being_evaluated_index = alg_event_number[alg_event_number==alg_event_during_not_during_gt].index
      if  ((all(hit[alg_event_being_evaluated_index]==0)) &
          (all(false_positive[alg_event_being_evaluated_index]==0)) &
          (all(miss[alg_event_being_evaluated_index]==0))):
          #false_positive_counter = false_positive_counter +1
          #false_positive_type_2[alg_event_being_evaluated_index] = 1    
          None
          None
    '''
    if label_names is None:
      event_type_name = str(event_type)
    else:
      event_type_name = label_names[i_label_name]
    output_dict = {event_type_name+'_hit':hit,
                   event_type_name+'_false_positive':false_positive,
                   event_type_name+'_false_positive_type_2':false_positive_type_2,
                   event_type_name+'_miss':miss,
                   event_type_name+'_hit_count':hit_counter,
                   event_type_name+'_false_positive_count':false_positive_counter,
                   event_type_name+'_miss_count':miss_counter,
                   event_type_name+'_hits_IoU':IoU_array,
                   event_type_name+'_f1_score':event_level_f1_score(hit_counter,false_positive_counter,miss_counter)}
  
    output.append(output_dict)
    output_2.update({event_type_name:output_dict})
    
  overall_hit =np.zeros(len(gt))
  overall_false_positive = np.zeros(len(gt))
  overall_false_positive_2 = np.zeros(len(gt))
  overall_miss =np.zeros(len(gt))
  overall_hit_counter =0
  overall_false_positive_counter = 0
  overall_miss_counter = 0
  overall_IoU = np.zeros(len(gt))

  if label_names is None:
    event_type_names = [str(ii) for ii in event_types]
  else:
    event_type_names = label_names     
    
  for ii,event_type_name in zip(output,event_type_names):
    print('event_type_name: ',event_type_name)
    overall_hit[ii[event_type_name+'_hit']!= 0] = 1
    overall_false_positive[ii[event_type_name+'_false_positive']!=0] = 1
    overall_miss[ii[event_type_name+'_miss']!= 0] = 1
    overall_false_positive_2[ii[event_type_name+'_false_positive_type_2']!= 0] = 1
    overall_hit_counter = overall_hit_counter + ii[event_type_name+'_hit_count']
    overall_false_positive_counter = overall_false_positive_counter + ii[event_type_name+'_false_positive_count']
    overall_miss_counter = overall_miss_counter + ii[event_type_name+'_miss_count']
    overall_IoU= overall_IoU+ii[event_type_name+'_hits_IoU']
  
  #overall_hit[np.where(overall_false_positive_2==1)]=0 # Set all hit samples during  false positives of type 2  to be 0

  output_dict = {}
  output_dict={'overall_hit':overall_hit,
                 'overall_false_positive':overall_false_positive,
                 'overall_false_positive_type_2':overall_false_positive_2,
                 'overall_miss':overall_miss,
                 'overall_hit_count':overall_hit_counter,
                 'overall_false_positive_count':overall_false_positive_counter,
                 'overall_miss_count':overall_miss_counter,
                 'overall_hits_IoU':overall_IoU,
                 'overall_f1_score':event_level_f1_score(overall_hit_counter,
                                                         overall_false_positive_counter,
                                                         overall_miss_counter)}

  # output_2 is the same as output except it is in a dict instead of a list
  output.append(output_dict)
  output_2.update({'overall':output_dict})
  return output_2

def getEventStartsAndEnds(labelled_event):
  '''
      Helper function to get the starts and ends of events from a sample level array of labels
      Parameters:
      -----------
            labelled_event  : a numpy array of labels at a sample level
                              E.g:
                                  [0,0,0,1,1,1,3,3,3,0,0,0,2,2,2,2]

    
        Returns:
        --------
            event_numbers : An array the size of the labelled_event that contains the event number at each index. 
            event_start   : An array that contains the index of when an event starts.
            event_end     : An array that contains the index of when an event ends.
  '''
  event_start = np.where(np.roll(labelled_event,1)!=labelled_event)[0]
  if event_start.size == 0:
      return 0,None,None
  if event_start[0] != 0:
      event_start = np.insert(event_start,0,0) # Add start of the first event in the case where the first and alst even are the same
  event_end = np.roll(event_start,-1)
  event_end[-1] = len(labelled_event) # Add the end of the last event (0 indexing should get yeeted) (threw away -1)
  event_end = event_end-1 # shift everything down 1 index then an event ends right before another starts
  event_numbers = np.zeros(len(labelled_event))
  
  for event_number,(start,end) in enumerate(zip(event_start,event_end),start=1):
    if start == end:
      event_numbers[start] = event_number
    else:
      event_numbers[start:end+1] = event_number
  return event_numbers, event_start,event_end 

if __name__=='__main__':
    '''
        Examples of how to run the event level matching. alg is your algorithms labels and gt is the ground truth labels. 
    '''
    alg = [0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,1,1,1,1,0,0,0,2,1,1,1,1,1]
    gt = [0,0,0,1,1,1,1,1,1,0,0,0,2,2,2,2,2,3,3,3,3,3,1,1,1,1,0,0,0,0,1,1,1,0,1]
    container = eventLevelMatchingIoU(gt,alg)

    container_1 = eventLevelMatchingIoU(gt,alg,label_names = None,event_types = [0,3])
    container_2 = eventLevelMatchingIoU(gt,alg,label_names = ['a','b','c'],event_types = [0,3,2])

# Structured Procrastination with Confidence
This is an implementation of Structured Procrastination with Confidence, an algorithm configuration procedure described in the paper [Procrastinating with Confidence: Near-Optimal, Anytime, Adaptive Algorithm Configuration](https://arxiv.org/pdf/1902.05454.pdf).

#### Abstract 
Algorithm configuration methods optimize the performance of a parameterized heuristic algorithm on a given distribution of problem instances. Recent work introduced an algorithm configuration procedure ("Structured Procrastination") that provably achieves near optimal performance with high probability and with nearly minimal runtime in the worst case. It also offers an _anytime_ property: it keeps tightening its optimality guarantees the longer it is run. Unfortunately, Structured Procrastination is not _adaptive_ to characteristics of the parameterized algorithm: it treats every input like the worst case. Follow-up work ("LeapsAndBounds") achieves adaptivity but trades away the anytime property. This paper introduces a new algorithm, "Structured Procrastination with Confidence", that preserves the near-optimality and anytime properties of Structured Procrastination while adding adaptivity. In particular, the new algorithm will perform dramatically faster in settings where many algorithm configurations perform poorly. We show empirically both that such settings arise frequently in practice and that the anytime property is useful for finding good configurations quickly.

#### Requirements
Python 2.7, pickle (for saving results), matplotlib (for generating plots).

#### Experimental Setup
The saved runtimes, simulated environment, and general experimental framework are all taken from this [repo](https://github.com/deepmind/leaps-and-bounds), which is an implementation of [LeapsAndBounds](https://arxiv.org/pdf/1807.00755.pdf), another algorithm configuration procedure (see also [CapsAndRuns](http://proceedings.mlr.press/v97/weisz19a/weisz19a-supp.pdf), a followup work).   


#### Running the Code
Download the repo and unzip the data (``measurements.dump``) into the root folder. There are three configuration procedures to be run: ``structured_procrastination``, ``structured_procrastination_confidence``, and ``leapsandbounds``. They are run by calling 
```
python <name-of-procedure>.py
``` 
at the command line. For example, calling
```
python structured_procrastination_confidence.py
``` 
will run the structured procrastination procedure. 


To produce the main plot (Figure 2 from the paper) call
```
python plot_results.py
``` 


#### Runtime Variation in Practice
To produce the plots demonstrating runtime variation in practice call  
```
python3 runtime_variation.py
``` 
Note that this requires Python 3.6. 
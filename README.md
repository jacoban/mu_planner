# mixed_uncertainty_planner

Developed and tested with Python 3.6.

Full instructions to replicate the results in the paper:

- Install the following packages:
    - pyro-ppl (https://pyro.ai/)
    - shapely (https://pypi.org/project/Shapely/)
    - numpy, scipy, matplotlib

- Create a 'log' folder inside the project root

- Launch one of the experiments scripts from inside the "scripts" folder (one is MG-HU, two is MG-SU, three is HG-HU, four is BL):
    
```
cd scripts
python experiments_one.py # two, three, four
```

- To get the Table (divided in three parts, for different CPs):

```
cd scripts
python get_results_tables --c_prob=0.01 # 0.1, 0.25
```

- Note: it looks like the numpy random generator changed between vers. 1.16 and 1.18. 
The results in the paper were obtained with vers. 1.18.

# Quantifying Faulty Assumptions in Heterogeneous Mutliagent Systems

In the embbedded video we demonstrate our running example in the CARLA autonomous simulator. When the two agents perceive the world differently the assumed strategies will cause a mismatch and create the potential for a collision.

{% include youtube_carla.html %}


## How to run

When running the code, ensure that the Storm model checker (https://www.stormchecker.org/) and its Stormpy bindings (https://moves-rwth.github.io/stormpy/).

For the running example with the street and pedestrain, run:

          samplingprism_mdp.py
  
For the grid-based experiments, run:

          car_grid.py

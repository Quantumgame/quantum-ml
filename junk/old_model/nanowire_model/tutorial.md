How to use the nanowire-model 

1. Instantiate the physics class at the beginning of the program. The parameters required are passed as a dict. See physics.py for the list of parameters. 

Note: There is no need to to define the object over and again. The self.V param can be set over different potential profiles.

2. Instantiate the markov class.
3. Find the num_dot estimate.
4. Use the fixed mu solver and num_dot estimate to find the starting node of the graph.
5. Generate the graph.

# Simulating-Materials-at-Constant-Pressure-Using-Homogeneous-Coordinates
To run code from equilibrium state for best results
1) Run code at desired target pressure using Routine 1; initial lattice structure and Boltzmann distribution for initial veclocity with W=10e6 until thermalisation.
2) Then run simulation again using Routine 2; set lamfin equal to the final value of lambda from first run, load the final positons into Is and final atomic velcoties into Ivs. This is all for any value of W (note that W affects dynamics).

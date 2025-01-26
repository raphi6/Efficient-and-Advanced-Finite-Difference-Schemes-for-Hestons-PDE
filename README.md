# Dissertation
Solving Heston's partial differential equation by finite difference methods.

# Code

Introduction :

	This program provides an implementation as described in the report, it prices European Call options under
	the Heston Stochastic Volatility model by Finite Difference Methods to solve the PDE. This is done by:
 	
  	- Discretising over non-uniform grids,
   	- Utilising "the ubiquitous Kronecker product",
    - Applying ADI time stepping schemes.
     	
      	There are also functions to plot results and analysis.

How to Use my Program :

    	Recommend to use PyCharm Community edition with Python 3.9
	
	- Extract the zip file
	- Download the dependencies from the import section
	- Open FDMheston.py. Now inside there are multiple options, you can:
		
		- run the function FDMheston() on its own with starting parameters.
  
			- change the arguments for Heston parameters
			- change the following two parameters for initial S0 and v0: S_current = ... V_current = ... 
	
    - activate plotting of surface by setting, plot_all = True
		- plot discretisation errors by setting either, plot_spatial = True  OR  plot_temporal = True
		- enable damping for Douglas by setting damping_Do = True
		- enable damping for Craig-Sneyd by setting damping_CS = True
		- print computation time by uncommenting the following:
			
			#print("Computation Time (ADI loop):", time)
			#print("Computation Time 2 (A matrix):", time2)
			#print("Computation time (All): ", time4)

		- plot the A matrices by uncommenting the following, this reveals their mentioned structure:
					
				# plot_sparse_matrix(A_0, 'A_0')
				# plot_sparse_matrix(A_1, "A_1")
				# plot_sparse_matrix(A_2, "A_2")
				# plot_sparse_matrix(A, "A")

# =========================================================================== #
# Compliant julia 1.x

# Using the following packages
using JuMP, GLPK
using LinearAlgebra
using Plots

include("loadSPP.jl")
include("setSPP.jl")
include("getfname.jl")

include("greedy_search.jl")
include("descent1.jl")
include("descent2.jl")
include("grasp.jl")
include("reactivegrasp.jl")

include("experiment.jl")

include("genetic.jl")





# =========================================================================== #

# Loading a SPP instance
println("\nLoading...")

#==============================DM3============================================#

#=Tous les paramètres sont à régler dans les fonctions simulationGA et plot_experimentGA
    situés dans le fichier experiment.jl(tout en bas)
   =#

#Pour tester sur toutes les instances de type pb_100rnd du dossier Data1/ 
#simulationGA("Data1/")

#Pour tester sur une instance particulière avec sortie graphique
fname="Data1/pb_100rnd0200.dat"
plot_experimentGA(fname)

# =============================DM3============================================== #



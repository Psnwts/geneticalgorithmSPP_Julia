
using LinearAlgebra
#using PyPlot

include("grasp.jl")
include("descent2.jl")
include("greedy_search.jl")
include("descentechatterbox.jl")




function plotRunGrasp(iname,zinit, zls, zbest)

    ion()
    figure("Examen d'un run",figsize=(6,6)) # Create a new figure
    title("GRASP-SPP | \$z_{Init}\$  \$z_{LS}\$  \$z_{Best}\$ | " * iname)
    xlabel("Itérations")
    ylabel("valeurs de z(x)")
    ylim(0, maximum(zbest)+2)

    nPoint = length(zinit)
    x=collect(1:nPoint)
    xticks([1,convert(Int64,ceil(nPoint/4)),convert(Int64,ceil(nPoint/2)), convert(Int64,ceil(nPoint/4*3)),nPoint])
    plot(x,zbest, linewidth=2.0, color="green", label="meilleures solutions")
    plot(x,zls,ls="",marker="^",ms=2,color="green",label="toutes solutions améliorées")
    plot(x,zinit,ls="",marker=".",ms=2,color="red",label="toutes solutions construites")
    vlines(x, zinit, zls, linewidth=0.5)
    legend(loc=4, fontsize ="small")

end


# Simulation d'une experimentation numérique  --------------------------


function simulationgrasp(foldername)
    fnames=readdir(foldername)

    #Paramètres:
    nbInstances       =  length(fnames)
    nbRunGrasp        =  10   # nombre de fois que la resolution GRASP est repetee
    nbIterationGrasp  =  200  # nombre d'iteration que compte une resolution GRASP
    alpha             =  0.75

    zmax  = zeros(Float64,nbInstances); zmax[:].=typemin(Int64)
    zmoy  = zeros(Float64, nbInstances) 
    zmin  = zeros(Float64,nbInstances) ; zmin[:].=typemax(Int64)
    tmoy  = zeros(Float64, nbInstances)  
    
    println("Experimentation GRASP-SPP avec :")
    println("  nbInstances       = ", nbInstances)
    println("  nbRunGrasp        = ", nbRunGrasp)
    println("  nbIterationGrasp  = ", nbIterationGrasp)
    println("  alpha             = ", alpha)
    println(" ")
    cpt = 0


    for instance = 1:nbInstances

        C,A = loadSPP(foldername*fnames[instance])
        print("  ",fnames[instance]," : ")
        for runGrasp = 1:nbRunGrasp
        

            start = time() # demarre le compteur de temps
            zbetter,zworse,zavg= graspSPPtest(alpha,nbIterationGrasp,A,C)
            tutilise = time()-start # arrete et releve le compteur de temps
            cpt+=1; print(cpt%10)
            
            # mise a jour des resultats collectes
            
            zmax[instance] = max(zmax[instance],zbetter)
            zmin[instance] = min(zmin[instance],zworse)
            zmoy[instance] += zavg
            tmoy[instance] +=tutilise
           
        end #run

        zmoy[instance] = zmoy[instance] / nbInstances
        tmoy[instance] = tmoy[instance] / nbInstances
        println(" ")
    end #instance
    
    #Affichage

    print("Instance")
    print("\t\t\t")
    print("Avg CPUt(seconds)")
    print("\t")
    print("min")
    print("\t\t")
    print("moyn")
    print("\t\t")
    print("max")
    println()

    
    for i=1:nbInstances
        print(fnames[i])
        print("\t\t")
        print(round(tmoy[i],digits=3))
        print("\t\t\t")
        print(zmin[i])
        print("\t\t")
        print(round(zmoy[i],digits=3))
        print("\t\t")
        print(zmax[i])
        println("")

    end

end


function simulationreactivegrasp(foldername)
    fnames=readdir(foldername)

    nbInstances       =  length(fnames)
    
    #Paramètres:
    nbRunGrasp        =  10   # nombre de fois que la resolution GRASP est repetee
    nbIterationGrasp  =  500  # nombre d'iteration que compte une resolution GRASP
    vectalpha         = [0.8,0.95,0.75,0.99]
    Nalpha            =   100

    zmax  = zeros(Float64,nbInstances); zmax[:].=typemin(Int64)
    zmoy  = zeros(Float64, nbInstances) 
    zmin  = zeros(Float64,nbInstances) ; zmin[:].=typemax(Int64)
    tmoy  = zeros(Float64, nbInstances)  

    println("Experimentation REACTIVE GRASP-SPP avec :")
    println("  nbInstances       = ", nbInstances)
    println("  nbRunGrasp        = ", nbRunGrasp)
    println("  nbIterationGrasp  = ", nbIterationGrasp)
    println("  vectalpha         = ",vectalpha)
    println("  Nalpha            = ", Nalpha)
    println(" ")
    cpt = 0


    for instance = 1:nbInstances

        C,A = loadSPP(foldername*fnames[instance])
        print("  ",fnames[instance]," : ")
        for runGrasp = 1:nbRunGrasp
        

            start = time() # demarre le compteur de temps
            zbetter,zworse,zavg=reactivegrasp(A,C,vectalpha,Nalpha,nbIterationGrasp)
            tutilise = time()-start # arrete et releve le compteur de temps
            cpt+=1; print(cpt%10)
            
            # mise a jour des resultats collectes
            
            zmax[instance] = max(zmax[instance],zbetter)
            zmin[instance] = min(zmin[instance],zworse)
            zmoy[instance] += zavg
            tmoy[instance] +=tutilise
           
        end #run

        zmoy[instance] = zmoy[instance] / nbInstances
        tmoy[instance] = tmoy[instance] / nbInstances
        println(" ")
    end #instance
    
    #Affichage

    print("Instance")
    print("\t\t\t")
    print("Avg CPUt(seconds)")
    print("\t")
    print("min")
    print("\t\t")
    print("moyn")
    print("\t\t")
    print("max")
    println()

    
    for i=1:nbInstances
        print(fnames[i])
        print("\t\t")
        print(round(tmoy[i],digits=3))
        print("\t\t\t")
        print(zmin[i])
        print("\t\t")
        print(round(zmoy[i],digits=3))
        print("\t\t")
        print(zmax[i])
        println("")

    end

end

function simulationgreedy(str)
    println("Loading....")
    
    fnames=readdir(str)

    solutionsconstr=Array{Float64}(undef,size(fnames)[1])
    admissibles=Array{Bool}(undef,size(fnames)[1])
    tconsts=Array{Float64}(undef,size(fnames)[1])

    Solutions_amels=Array{Float64}(undef,size(fnames)[1])
    tamels=Array{Float64}(undef,size(fnames)[1])
    for i in 1:size(fnames)[1]
        C, A = loadSPP(str*fnames[i])
        

        println(fnames[i])
        x0=Array{Integer}(undef,size(C)[1])
        
        greedystart = time()
        x0=greedy_search(A,C)
        tconstruct = time()-greedystart

        Zcons=LinearAlgebra.dot(x0,C)
        admi = isAdmissible(A,x0)
        
        descentstart = time()
        solamel = simple_descentcb(A,x0,C)
        tamel = time() - descentstart

        Zamel=LinearAlgebra.dot(solamel,C)
        solutionsconstr[i] = Zcons
        admissibles[i] = admi
        Solutions_amels[i] = Zamel

        tconsts[i] = tconstruct
        tamels[i] = tamel
    end


    #Affichage:

    print("fname\t\t\t\t")
    print("Zcons\t")
    print("isAdmissible\t")
    print("tconst\t\t")
    print("Zamel\t\t")
    println("tamel\t") 
    for i in 1:size(fnames)[1]

        print(fnames[i])
        print("\t\t")
        print(solutionsconstr[i])
        print("\t")
        print(admissibles[i])
        print("\t\t")
        print(round(tconsts[i],digits=3))
        print("\t\t")
        print(Solutions_amels[i])
        print("\t\t")
        println(round(tamels[i],digits=3))
        
        
    end
    
end


#Ajouts DM3:

function plot_experimentGA(instance)
    #Permet d'effectuer une expérimentation graphique sur plusieurs instances de GA

    C,A=loadSPP(instance)

    #Paramètres d'une seule instance de GA
    alpha        = 0.80 #paramètre de la greedyRandomizedConstruction lors de l'initialisation de la population
    nIndividuals = 100  #nombre d'individu dans une population 
    nIterations  = 100  #nombre de fois que la population est mise à jour dans l'algorithme génétique
    Pc           = 0.99 #Probabilité de croisement
    Pm           = 0.03 #Probabilité de mutation
    nGenerations = 40   #Nombre de generations
    isrepair     = false
    
    nbRunGA      = 10 # Nombre de fois que l'algorithme genetique est lancé
    moyndesZmoy  = zeros(Float64,nGenerations) # Tableau contenant les moyennes des Zmoy de toutes les instances de GA
    moyndesZmax  = zeros(Float64,nGenerations) # Tableau contenant les moyennes des Zmax de toutes les instances de GA

    
    cpt=0

    print(" ",instance,": ")
    for i in 1:nbRunGA
        
        generations,eval_gens,Zmoy,Zmax=genetic_algorithm(A,C,alpha,nIndividuals,nIterations,Pc,Pm,nGenerations,isrepair)
        cpt+=1;print(cpt%(nbRunGA+1)," ")

        moyndesZmoy     = map(+,moyndesZmoy,Zmoy)
        moyndesZmax     = map(+,moyndesZmax,Zmax)

        
    end

    for i in 1:nGenerations
        moyndesZmoy[i] = moyndesZmoy[i] / nbRunGA
        moyndesZmax[i] = moyndesZmax[i] / nbRunGA
    end

    #Affichage graphique:

    x=range(1,nGenerations)
    y1=moyndesZmoy
    y2=moyndesZmax
    str="AG|"*instance*"|"*"taille population= "*string(nIndividuals)*" |nGeneration= "*string(nGenerations)*"|Pc= "*string(Pc)*"|Pm= "*string(Pm)*"|nbRunGA= "*string(nbRunGA)
    p = plot([x x],[y1 y2],label=["Zmoyen" "Zmax"],linewidth=3,xlabel="Generations",ylabel="Average fitness")
    plot!(title=str,titlefont=7)
    display(plot(p))

end

function simulationGA(foldername)
    fnames=readdir(foldername)

    nbInstances       =  length(fnames)
    #Paramètres d'une seule instance de GA
    alpha        = 0.80  #paramètre de la greedyRandomizedConstruction lors de l'initialisation de la population
    nIndividuals = 10   #nombre d'individu dans une population 
    nIterations  = 10   #nombre de fois que la population est mise à jour dans l'algorithme génétique
    Pc           = 0.99  #Probabilité de croisement
    Pm           = 0.03   #Probabilité de mutation
    nGenerations = 10     #Nombre de generations

    #Paramètres:
    nbRunGA        =  10   # nombre de fois que la resolution GRASP est repetee



    zmax         = zeros(Float64,nbInstances); zmax[:].=typemin(Int64)
    lastgenzmoy  = zeros(Float64,nbInstances) 
    tmoy         = zeros(Float64,nbInstances)  

    println("Experimentation GENETIC ALGORITHM-SPP avec :")
    println("  nbInstances    = ",      nbInstances)
    println("  nbRunGA        = ",          nbRunGA)
    println("  NbGenerations  = ",     nGenerations)
    println("  NbIndividuals  = ",     nIndividuals)
    println("  Pc             = ",               Pc)
    println("  Pm             = ",               Pm)
    println(" ")
    


    for instance = 1:nbInstances

        C,A = loadSPP(foldername*fnames[instance])
        print("  ",fnames[instance]," : ")
        cpt = 0
        
        #Initialisation à chaque instance:
        moyndesZmoy  = zeros(Float64,nGenerations) # Tableau contenant les moyennes des Zmoy de toutes les instances de GA
        moyndesZmax  = zeros(Float64,nGenerations) # Tableau contenant les moyennes des Zmax de toutes les instances de GA
        for runGA = 1:nbRunGA
        

            start = time() # demarre le compteur de temps
            generations,eval_gens,Zmoy,Zmax=genetic_algorithm(A,C,alpha,nIndividuals,nIterations,Pc,Pm,nGenerations)
            tutilise = time()-start # arrete et releve le compteur de temps
            cpt+=1; print(cpt%(nbRunGA+1)," ")
            
            # mise a jour des resultats collectes

            moyndesZmoy     = map(+,moyndesZmoy,Zmoy)
            moyndesZmax     = map(max,moyndesZmax,Zmax)
            tmoy[instance] +=tutilise
           
        end #runGA

        for i in 1:nGenerations
            moyndesZmoy[i] = moyndesZmoy[i] / nbRunGA 
        end
        lastgenzmoy[instance] = moyndesZmoy[length(moyndesZmoy)] #Zmoyen de la dernière génération
        zmax[instance]        = maximum(moyndesZmax)
        tmoy[instance]        = tmoy[instance] / nbInstances
        println(" ")
    end #instance
    
    #Affichage

    print("Instance")
    print("\t\t\t")
    print("Avg CPUt(seconds)")
    print("\t")
    print("lastgenmoyn")
    print("\t\t")
    print("max")
    println()

    
    for i=1:nbInstances
        print(fnames[i])
        print("\t\t")
        print(round(tmoy[i],digits=3))
        print("\t\t\t")
        print(round(lastgenzmoy[i],digits=3))
        print("\t\t\t")
        print(round(zmax[i],digits=3))
        println("")

    end
end


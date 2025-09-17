using LinearAlgebra
using Plots

include("greedy_search.jl")
include("grasp.jl")


function evaluation(x,C)
    #fontion fitness
    #permet l'évaluation des individus
    z = LinearAlgebra.dot(x,C)
    return z
end 

function compute_utility(A,C)
    #fonction permettant de calculer l'utilité associée à instance particulière (Ex: pb_100rnd0100) 

    # tableau contenant le nombre de fois où chaque activité i apparait dans les 
    # ressources
    occurences = Vector{Float64}(undef,size(C)[1])
    
    for i in 1:size(occurences)[1]
        occurences[i] = 0
    end

    for i in 1:size(A)[1] 
        for j in 1:size(A)[2] # nombre de colonnes
            if( A[i,j] != 0)
                occurences[j]+=1
            end
        end
    end

    utility=Array{Float64}(undef,size(C)[1])


    for i in 1:size(C)[1]
        utility[i] = (C[i] / occurences[i])
    end

    return utility
end

function init_population(A,C,alpha,nIndividuals)
    #fonction permettant l'initialisation d'une population via une greedyRandomizedConstruction
    
    population  = []
    evaluations = zeros(Float64,nIndividuals)

    for i in 1:nIndividuals
        p              = greedyRandomizedConstruction(A,C,alpha)
        e              = LinearAlgebra.dot(p,C) 
        evaluations[i] = e
        push!(population,p)
    end

    #print(population,evaluations)
    return population,evaluations

end

function selection_roulette(population,evaluations,nbreDeLancements=2)

    i=0
    s=0

    selected_elts     = [] # tableau stockant les éléments aléatoirement piochés
    selected_indexes  = [] # tableau stockant les indices des éléments aléatoirement piochés
    sumpevaluations   = sum(evaluations) # la somme des valeurs des fitness de la population
    sortedevaluations = sort(evaluations) # tableau dans lequel les fonctions fitness de la populations sont triées par ordre croissant
    sortedindexes     = indexin(sortedevaluations,evaluations) # tableau dans lequel les indices du tableau population 
                                                               # sont triés par ordre croissant selon les valeurs de leurs fitness
    
    while(i<nbreDeLancements)#nombre de fois que la roue est lancée
        
        pioche = rand(0:sumpevaluations)
        for j in 1:size(evaluations)[1]
            s += sortedevaluations[j]
            if(pioche < s)
                push!(selected_elts,population[j])
                push!(selected_indexes,sortedindexes[j])
            end
        end
        i+=1
    end
    parent1 = selected_elts[1]
    indice1 = selected_indexes[1]
    parent2 = selected_elts[2]
    indice2 = selected_indexes[2]
    return parent1,parent2,indice1,indice2
end

function selection_elite(population,evaluations)
    
    max1 = argmax(evaluations)
    temp = evaluations[max1]
    evaluations[max1] = 0
    max2=argmax(evaluations)
    evaluations[max1] = temp

    p1=population[max1]
    p2=population[max2]
    
    
    return p1,p2,max1,max2
end

function tournament(o1,o2,o3,o4,indice1,indice2,C)
    #L'objectif de cette fonction est de faire affronter deux parents et deux enfants et prend à chaque fois le meilleur des deux.
    #println(C)
    opp1=evaluation(o1,C)
    opp2=evaluation(o2,C)
    
    if(opp1 > opp2)
        winner1     = o1
        loser1      = o2
        indiceloser = indice2
    else
        winner1      = o2
        loser1       = o1
        indiceloser  = indice1
    end

    opp3=evaluation(o3,C)
    opp4=evaluation(o4,C)
    
    if(opp3 > opp4)
        winner2 = o3
    else
        winner2 = o4
    end

    return winner1,winner2,loser1,indiceloser
end

function crossover(p1,p2)
    #fonction de crossover avec une seule coupe
    l=length(p1)
    e1=Array{Integer}(undef,l)
    e2=Array{Integer}(undef,l)
    
    coupe=rand(1:l) #coupe aléatoire

    part11 = p1[1:coupe] #Le premier segment du parent 1
    part12 = p2[coupe+1:end]#le deuxième segment du parent 2
    
    e1 = vcat(part11,part12) #concatenation des deux segments donnant lieu à l'enfant 1

    part21 = p2[1:coupe]    #le premier segment du parent 2
    part22 = p1[coupe+1:end]#le deuxième segment du parent 1

    e2 = vcat(part21,part22) # concatenation des deux segments donnant lieu à l'enfant 2

    return e1,e2
end

function mutation(e)
    #prend une variable au hasard d'une solution et la met à 1
    #si la variable en question est déja mise à 1, on choisit une autre variable au hasard.
    l=length(e)

    while(true)
        randomflip = rand(1:l)
        if(e[randomflip] == 0)
            e[randomflip] = 1
            break
        end
    end

    return e
end

function repair(A,child,utility)
    #Tente de réparer un enfant non admissible
    #Cette fonction peut dégrader la valeur de la fonction fitness.
    #Afin de le réparer, nous parcourons l'individu et mettons les variables activées(mises à 1) à 0
    #Pour cela, nous créons un tableau des utilitées de chaque variables triées par ordre croissant
    #(cont.)puis nous mettant à 0 les variables selon cet ordre. Le but étant de réparer l'individu en dégradant le moins possible
    #(cont.)la valeur de sa fonction fitness.  
    
    product       = map(*, child, utility)
    sortedproduct = sort(product) #tableau contenant les utilités de chaque variable de l'individu triées par ordre croissant
    indexes       = indexin(sortedproduct,product) #tableau contenant les indices de chaque variables triés par ordre croissant des valeurs de leurs utilités

    for i in 1:size(indexes)[1]
        if(child[indexes[i]] != 0)
            child[indexes[i]] = 0 # met une variable à 0
            if(isAdmissible(A,child)) # test de l'admissibilté de l'individu modifié
                return child
            end
        end
    end
    return child
end

function generate_generation(A,C,utility,individuvide,populations,evaluations,nIterations,Pc,Pm,isrepair)
    i=0
    #Crée une seule génération d'individus à partir d'une population.
    #étape  1: sélétion de deux parent via la fonction selection_roulette()
    #étape  2: Effectue un crossover des deux parents selectioné via la fonction crossover()
    #étape  3: Effectue la procédure repair si elle est activée dans les paramètres. (elle ne l'est pas par défaut)
    #étape  4: met à jour la population par incrémentation via la fonction tournament()
    #étape  5: retour à l'étape 1 
    # Cette procédure est effectuée (nIterations) fois
    for i in 1:nIterations
    
        parent1,parent2,indice1,indice2=selection_roulette(populations,evaluations)

        p=rand()
        if(p < Pc)
            child1,child2=crossover(parent1,parent2)
        else

            continue
        end

        #mutation:
        p1 = rand()
        if(p1 < Pm)
            child1 = mutation(child1)
        end
            
        p2 = rand()
        if(p2 < Pm)
            child2 = mutation(child2)
        end
        
        #procédure repair:
        Zparent1 =evaluation(parent1,C) 
        Zparent2 =evaluation(parent2,C) 
        Zchild1  =evaluation(child1,C) 
        Zchild2  =evaluation(child2,C) 

        if(isrepair)
            child1  = repair(A,child1,utility)
            Zchild1 = evaluation(child1,C)
        end 

        #enfant 2
        if(isrepair)

            child2  = repair(A,child2,utility)
            Zchild2 = evaluation(child2,C)
        end 

        #Nous gardons l'enfant seulement si celui-ci est admissible et que sa valeur de Z est supérieur
        #(cont. ) à au moins celle d'un des deux parents. 
        flag1 = isAdmissible(A,child1) && (Zchild1 > Zparent1 || Zchild1 > Zparent2)
        flag2 = isAdmissible(A,child2) && (Zchild2 > Zparent1 || Zchild2 > Zparent2)
        
        if( !flag1 && !flag2)
            # kill les deux enfants.
            # Aucun des deux ne présente une valeur prometteuse
            continue 
        elseif( !flag1 || !flag2 )
            #Si un des deux enfants présente une valeur prometteuse
            if(flag1)

                winner1,winner2,loser1,indexOfParentSortant = tournament(parent1,parent2,child1,individuvide,indice1,indice2,C)     
                populations[indexOfParentSortant]           = winner2
                evaluations[indexOfParentSortant]           = evaluation(winner2,C)

            end
            if(flag2)
                winner1,winner2,loser1,indexOfParentSortant =tournament(parent1,parent2,child2,individuvide,indice1,indice2,C) 
                populations[indexOfParentSortant]           = winner2
                evaluations[indexOfParentSortant]           = evaluation(winner2,C)
            end
        elseif(flag1 && flag2)
            #Si les deux enfants présentent des valeurs prometteuses:
            winner1,winner2,loser1,indexOfParentSortant = tournament(parent1,parent2,child1,child2,indice1,indice2,C)
            populations[indexOfParentSortant] = winner2
            evaluations[indexOfParentSortant] = evaluation(winner2,C)

        end
    end

    return populations,evaluations
end


function genetic_algorithm(A,C,alpha,nIndividuals,nIterations,Pc,Pm,nGenerations,isrepair=false)
    #Crée nGenerations de populations.

    i=0
    generations = [] # tableau contenant les différentes populations de chaque génération
    eval_gens   = [] # tableau contenant les différentes valeurs fitness de chaque génération
    Zmoy        = zeros(Float64,nGenerations)
    Zmax        = zeros(Float64,nGenerations)
    valmax      = 0

    utility                 = compute_utility(A,C)
    individuvide            = zeros(Float64,size(C)[1]) #[0,0 ......,0]
    populations,evaluations = init_population(A,C,alpha,nIndividuals)
    str=""
    
    while(i < nGenerations)
        #println(Core.Typeof(A))
        #println(sizeof(C))
        population,evaluations=generate_generation(A,C,utility,individuvide,populations,evaluations,nIterations,Pc,Pm,isrepair)
        
        Zmoy[i+1] = sum(evaluations)/length(evaluations) #calcul de la moyenne
        valmax  = max(valmax,maximum(evaluations))
        Zmax[i+1] = valmax

        push!(generations,population)
        push!(eval_gens,evaluations)
        
        i+=1

    end
    return generations,eval_gens,Zmoy,Zmax
end


function plotting_genetic(Zmoyen,Zmax,NbGenerations)
    x=range(1,NbGenerations)
    y1=Zmoyen
    y2=Zmax
    p = plot([x x],[y1 y2],label=["Zmoyen" "Zmax"],linewidth=3)

    display(plot(p))
end

function plotting_(foldername)
    fnames = readdir(foldername)

    x=range(1,length(fnames))
    y1=[10.685,0.332,3.571,1.021,1.157,1.059,0.51,0.608,2.631,0.516,2.337,1.386]
    y2=[71.252,5.417,42.3,2.644,105.083,12.701,82.5,6.994,81.773,6.999,60.614,6.396]
    y3=[74.929,7.36,35.046,3.76,111.073,12.705,82.664,6.995,88.682,3.998,60.31,6.435]
    p = plot(fnames,[y1 y2 y3],label=["GA" "GRASP" "ReactiveGRASP"],linewidth=3,ylabel="Zbest-Zmoyen",xlabel="instances",xrotation=45,xtickfontsize=7)

    display(plot(p))
end


function plotting_two(foldername)
    fnames = readdir(foldername)

    x=range(1,length(fnames))
    y1=[368,30,195,15,622,64,495,37,453,37,306,23]
    y2=[363,30,195,14,627,64,495,37,443,37,306,23]
    y3=[368,32,195,15,633,64,495,37,456,37,306,23]
    p = scatter(fnames,[y1 y2 y3],label=["GA" "GRASP" "ReactiveGRASP"],linewidth=3,ylabel="Zbest",xlabel="instances",xrotation=45,xtickfontsize=7)

    display(plot(p))
end



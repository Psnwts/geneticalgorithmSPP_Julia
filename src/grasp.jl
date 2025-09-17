using  LinearAlgebra

include("descent2.jl")

#The greedy randomized construction
function greedyRandomizedConstruction(A,C,alpha)

    x0=Array{Integer}(undef,size(C)[1])
    c0=Array{Bool}(undef,size(C)[1])
   
    for i in 1:size(C)[1]
        x0[i] = 0
        c0[i] = true
    end

    occurences = Vector{Float64}(undef,size(C)[1])
    for i in 1:size(occurences)[1]
        occurences[i] = 0
    end

    for i in 1:size(A)[1] # nombre de lignes
        for j in 1:size(A)[2] # nombre de colonnes
            if( A[i,j] != 0)
                occurences[j]+=1
            end
        end
    end

    while(!(isempty(findall(c0)))) 

        utility=Array{Float64}(undef,size(C)[1])
        for i in 1:size(C)[1]
            utility[i] = (C[i] / occurences[i])*c0[i]
        end
        
        utility2 = [x for x in utility if x > 0] # tableau ne contenant que les éléments non nuls
        
        #Calcul de la valeur limite
        limit = minimum(utility2) + alpha * ( maximum(utility2) - minimum(utility2) )

        #construction de la restricted candidate list (RCL)
        rcl = findall(x -> x > limit || x ≈ limit, utility)
        
        #choix d'un élément ,au hasard, de la rcl
        e = rand(rcl)

        index = e[1]
        x0[e] = 1
        c0[e] = false

        lignes_saturees=findall(isequal(1),A[:,e])

        for i in 1:size(lignes_saturees)[1]
            elims = findall(isequal(1),A[lignes_saturees[i],:])
            for j in 1:size(elims)[1]
              
                if (elims[j] != e )
                    if(c0[elims[j]] == true)
                        c0[elims[j]] = false
                    end
                end
                
            end
            
        end
    end

    
    return x0
end


#fonction tirée du fichier experiment.jl mis à disposition pour le DM
function graspSPPtest(alpha, nbIterationGrasp,A,C)

    zconstruction = zeros(Int64,nbIterationGrasp)
    zamelioration = zeros(Int64,nbIterationGrasp)
    

    zbetter=0
    zworse=Inf

    S=0
    for i=1:nbIterationGrasp
        
        #I-greedyrandomizedconstruction(problem,alpha)
        xconst=greedyRandomizedConstruction(A,C,alpha)
        Zconst=LinearAlgebra.dot(xconst,C)
        
        #II-LocalSearchImprovement
        xamel= simple_descent2(A,xconst,C)
        Zamel=LinearAlgebra.dot(xamel,C)
        
        #solutions construites:
        zconstruction[i] = Zconst
        
        #améliorations
        zamelioration[i] = Zamel 

        #meilleur amélioration
        zbetter = max(zbetter,Zamel)

        #pire amélioration
        zworse = min(zworse,Zamel)

        #Somme des améliorations
        S+=Zamel
    end
    #Moyenne
    zmoy = S/nbIterationGrasp

    return zbetter,zworse,zmoy
end

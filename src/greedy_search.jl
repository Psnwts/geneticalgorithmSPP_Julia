function greedy_search(A,C)
    #I- Initialisations:

    # solution de base construite au fur et à mesure de la recherche gloutone
    x0=Array{Integer}(undef,size(C)[1])
    
    # Ensemble des candiats ( si c0[i] == true ==> candidat i séléctionable)
    #                       ( sinon ==> candidat i non séléctionable )
    c0=Array{Bool}(undef,size(C)[1])
   
    #Initialisation de c0 et de x0
    for i in 1:size(C)[1]
        x0[i] = 0
        c0[i] = true
    end

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

    #II- construction gloutone:

    while(!(isempty(findall(c0)))) 
        # c0 est vide lorsqu'il ne reste plus d'éléments séléctionables
        # c-à-d quand c[i] == false pour tout i de c0
        
        utility=Array{Float64}(undef,size(C)[1])

        #fonction d'utilité
        # lorsque i n'est plus séléctionable ==> utility[i] == (C/occurence) * false
        #                                    ==> utility[i] == 0
        for i in 1:size(C)[1]
            utility[i] = (C[i] / occurences[i])*c0[i]
        end
        
        index_max=argmax(utility)

        #On ajoute l'activité i à la solution
        x0[index_max] = 1

        #On retire l'activité i de l'ensemble des candidats
        c0[index_max] = false


        #élimination des conflits:

        lignes_saturees=findall(isequal(1),A[:,index_max])

        for i in 1:size(lignes_saturees)[1]
            elims = findall(isequal(1),A[lignes_saturees[i],:])
            for j in 1:size(elims)[1]
              
                if (elims[j] != index_max )
                    if(c0[elims[j]] == true)
                        c0[elims[j]] = false
                    end
                end
                
            end
            
        end

    end
    return x0
end

# Test de l'admissibilité de la solution construite
function isAdmissible(A,x0)
    for i in 1:size(A)[1]
        if (LinearAlgebra.dot(A[i,:],x0) > 1)
            return false
        end
    end
    return true
end
#roulette wheel selection algorithm
#Crédits: https://jamesmccaffrey.wordpress.com/2017/12/01/roulette-wheel-selection-algorithm/

#Sélection d'une valeur alpha compte tenu de sa probabilité Pk
function roulette_select(vectpb)

    cumul=0
    pb =rand()
    for i in 1:size(vectpb)[1]
        cumul+=vectpb[i]

        if(pb < cumul)
            return i
        end
    end

    last_elt = size(vectpb)[1]
    return  last_elt
end

function reactivegrasp(A,C,vectalpha,Nalpha,nbIterationGrasp)
    
    m = size(vectalpha)[1] # taille de [alpha1,alpha2,....,alpham]
    pk = Array{Float64}(undef,m)#vecteur des probabilité 

    for i in 1:size(pk)[1]
        pk[i] = 1 / m
    end

    #initialisations
    zconstruction = zeros(Int64,nbIterationGrasp)
    zamelioration = zeros(Int64,nbIterationGrasp)
    zbest = zeros(Int64,nbIterationGrasp)
    zworst = zeros(Int64,nbIterationGrasp)

    zbetter = 0
    zworse = Inf

    zavg = zeros(Float64,m)
    sums = zeros(Float64,m)
    iterations = zeros(Float64,m)

    alphas_selecteds = zeros(Int64,nbIterationGrasp)

    qk= zeros(Float64,m)
    
    S=0
    for i in 1:nbIterationGrasp
        
        #indice selectioné

        rand_index = roulette_select(pk)
        
        alphas_selecteds[i] = rand_index
        alpha = vectalpha[rand_index]

        #greedyRandomizedConstruction
        xconst=greedyRandomizedConstruction(A,C,alpha)
        Zconst=LinearAlgebra.dot(xconst,C)
        #local search improvement
        xamel= simple_descent2(A,xconst,C)
        Zamel=LinearAlgebra.dot(xamel,C)
        
        zconstruction[i] = Zconst
        zamelioration[i] = Zamel 
        
        zbetter = max(zbetter, zamelioration[i])
        zworse = min(zworse,zamelioration[i])
        
        zbest[i] = zbetter
        zworst[i] = zworse
        S+=Zamel
        
        sums[rand_index] += Zamel
        iterations[rand_index] +=1

        if( i % Nalpha == 0 )
            for i in 1:size(vectalpha)[1]
                #calcul de la moyenne ZAvgk 
                zavg[i] = sums[i] / iterations[i]
                #calcul de qk
                qk[i] = ( zavg[i] - zworse) / (zbetter - zworse)  
            end

            somme_qk=sum(qk)
            for i in 1:size(vectalpha)[1]
                pk[i] = qk[i]/somme_qk
                
            end
        end
        
    end

    zavg=S/nbIterationGrasp
    return zbetter,zworse,zavg
end


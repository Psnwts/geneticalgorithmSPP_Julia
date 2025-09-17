using LinearAlgebra

include("greedy_search.jl")

#I-test de l'admissibilité de chaque voisin améliorant
#=Nous avons choisi de tester l'admissibilité uniquement lors
de la mise à 1 des variables. Etant donné que la mise à zéro ne pose aucun
problème à l'admissibilité de la solution. =#

function isAdmissible_vsn(A,x,k)
    
    #On cherche les contraintes dans lesquels a(i,k) == 1
    lignes_modifies = findall(isequal(1),A[:,k])

    for i in 1:size(lignes_modifies)[1]
        if(LinearAlgebra.dot(A[lignes_modifies[i],:],x) > 1)
            return false   
        end
    end

    return true
end

#II- construction de solutions voisines à l'aide de k-p exchange

# 2-1 exchange
function deux_un_exchange(A,x0,C)
    xprime =copy(x0) # construction d'une solution voisine 
    Zconst = LinearAlgebra.dot(x0,C)
    for i in 1:size(x0)[1]
        if (x0[i] == 1)
            xprime[i]=0 # mise à zéro
        else
            continue
        end
        
        for j in 1:size(x0)[1]
            if (x0[j] == 1 && j != i)
                xprime[j]=0 #mise à zéro
            else
                continue
            end
            for k in 1:size(x0)[1]
                if (x0[k] == 0)
                    xprime[k]=1 # mise à un
                    Zamel  = LinearAlgebra.dot(xprime,C)
                    if(Zamel > Zconst) #tester si la solution voisine est améliorante
                        if(isAdmissible_vsn(A,xprime,k))
                            # test sur l'admissibilté de la solution voisine
                            return xprime,true
                        end
                    end
                else
                    continue
                end
                xprime[k] = 0 # réinitialisation
            end
            xprime[j]=1 #réinitialisation
        end
        xprime[i]=1# réinitialisation
    end
    return x0,false
end 


#1-1 exchange
function un_un_exchange(A,x0,C)
    xprime=copy(x0)# construction d'une solution voisine 
    Zconst = LinearAlgebra.dot(x0,C)
    
    for i in 1:size(x0)[1]
        if(x0[i]==1)
            xprime[i]=0
        else
            continue
        end
        for j in 1:size(x0)[1]
            if(x0[j]==0)
                xprime[j]=1

                Zamel  = LinearAlgebra.dot(xprime,C)
                if(Zamel > Zconst) 
                    if(isAdmissible_vsn(A,xprime,j))
                        return xprime,true
                    end
                end 
            else
                continue
            end
            xprime[j]=0 # réinitialisation

        end
        xprime[i]=1 # réinitialisation
    end
    return x0,false
end


#0-1 exchange
function zero_un_exchange(A,x0,C)

    Zconst=LinearAlgebra.dot(x0,C)
    xprime=copy(x0)
    for i in 1:size(x0)[1]
        if(x0[i] == 0)
            xprime[i] = 1
            Zamel =LinearAlgebra.dot(xprime,C)
            if(Zamel>Zconst)
                if(isAdmissible_vsn(A,xprime,i))
                    return xprime,true
                end
            end
        else
            continue
        end
        xprime[i] = 0
    end
    return x0,false
end

# Algorithme de descente:

function simple_descentcb(A,x0,C)
    println("Amelioration par recherche local de type descent simple")
    xamel=copy(x0)

    phrase2_1=""
    phrase1_1=""
    phrase0_1=""
    
    #= flagi == false avec (i=1,2,3) quand on ne trouve plus de solution voisine
     améliorante et admissible =#
    flag1=true
    flag2=true
    flag3=true

    print("2-1: ")
    while(flag1)
        xamel,flag1=deux_un_exchange(A,xamel,C)
        phrase2_1*="x"
        print(phrase2_1)
    end
    println()

    print("1-1: ")
    while(flag2)
       
        xamel,flag2=un_un_exchange(A,xamel,C)
        phrase1_1*="x"
        print(phrase1_1)

    end
    println()

    print("0-1: ")
    while(flag3)

        xamel,flag3=zero_un_exchange(A,xamel,C)
        phrase0_1*="x"
        print(phrase0_1)

    end
    println()
    
    return xamel
end
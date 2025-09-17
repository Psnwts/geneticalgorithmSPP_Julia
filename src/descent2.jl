using LinearAlgebra

include("greedy_search.jl")

function isAdmissible_vsn(A,x,k)

    lignes_modifies = findall(isequal(1),A[:,k])

    for i in 1:size(lignes_modifies)[1]
        if(LinearAlgebra.dot(A[lignes_modifies[i],:],x) > 1)
            return false   
        end
    end

    return true
end

function un_un_exchange(A,x0,C)
    xprime=copy(x0)
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
            xprime[j]=0

        end
        xprime[i]=1
    end
    return x0,false
end


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

function simple_descent2(A,x0,C)
    xamel=copy(x0)

    flag1=true
    flag2=true

    while(flag1)
        xamel,flag1=un_un_exchange(A,xamel,C)
    end

    while(flag2)
        xamel,flag2=zero_un_exchange(A,xamel,C)
    end

    
    return xamel
end
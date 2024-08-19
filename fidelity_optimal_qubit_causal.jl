using LinearAlgebra, MKL
using Convex, SCS
using Combinatorics
using QuantumInformation
using SparseArrays

function projector_s_s(N=1, atol=1e-3)
    
    """
        Input: N - the number of copies of the given measurement
               atol - tolerance level used to numerically create 
               a projection matrix with tested default vaule 10^(-3)

        Output: The projector onto commutator subspace.

        Note: Take matrix X. Define |Y>> := projector_s_s(N) |X>>.
               Then, [Y,U^{⊗N+1}] = 0 for all qubit unitary U.
    """

    # P is the projector created on 2 x (N+1) = 2N+2 subsystems
    P = zeros(2^(2N+2), 2^(2N+2))
    idx = CartesianIndices(Tuple(fill(2, N+1)))

    # P |ω_π>> = |ω_π>> for any system permutation matrix ω_π defined on N+1 systems
    for perm in permutations(1:N+1)
        x = zeros(fill(2, 2N+2)...)
        for k1 in idx
            k0 = CartesianIndex(Tuple(k1)[perm])
            x[k0, k1] = 1
        end
        x = vec(x)
        P += x * x'
    end

    # P is positive semidefinite matrix defined on the space spanned by |ω_π>>. 
    # It has to be normalized (eigenvalues 0 and 1) 
    F = eigen(P)
    Λ = Diagonal([x > atol ? 1 : 0 for x in F.values])
    P0 = F.vectors * Λ * F.vectors'
    sparse(P0)
end

function Qpartialtrace(Q, tuple)

    """
        Input: Q - a matrix of size 2^N x 2^N, where N is some integer
               tuple - binary tuple of size N

        Output: Partial trace of Q over qubit spaces X_j such that tuple[j] = 0.
    """
    
    N = Int(log2(size(Q, 1)))
    M = N
    for j=N:-1:1
        if tuple[j] == 0
            Q = partialtrace(Q, j, fill(2, M))
            M -= 1
        end
    end
    return Q
end

function calc_causal(N)

    """
        Input: N - the number of copies of the given measurement

        Output: The maximal value of the fidelity function F_d(N) for d = 2
                and N copies of given von Neumann measurement.  

        Note: We use SDP program defined in the paper as Program II. Additionally, we
              use two simplifications explained in Section IV for the case d = 2.
    """

    # P is the projector such that P |ω_π>> = |ω_π>> for any 
    # system permutation matrix ω_π defined on N+1 qubit systems
    P = projector_s_s(N)

    # Lijs is a list of 2^(N+1) operators L_{i,j} that operate on the 
    # space of size 2^(N+1)
    Lijs = [ComplexVariable(2^(N+1), 2^(N+1)) for _=1:2^(N+1)]

    # Lij ≥ 0
    constraints = [Lij in :SDP for Lij in Lijs]

    # [Lij, \bar(U) ⊗ U^(⊗N)] = 0, were we equivalently assume
    # [Lij, U^(⊗N+1)] = 0 by using projector P
    constraints += [P * vec(Lij) == vec(Lij) for Lij in Lijs]

    Lijs = reshape(Lijs, fill(2, N+1)...)

    # Partial sum of Lijs
    SumLijs = [sum(Lijs[j, i] for i=1:2) for j ∈ CartesianIndices(Tuple(fill(2, N)))]
    SumLijs = reshape(SumLijs, fill(2, N)...)

    # Network constraint
    constraints += [SumLijs[j] == partialtrace(SumLijs[j], N+1, fill(2, N+1)) ⊗ I(2)/2 for j=1:2^N]


    SumLijs = [partialtrace(SumLij, N+1, fill(2, N+1))/2 for SumLij ∈ SumLijs]
    SumLijs = reshape(SumLijs, fill(2, N)...)

    # Trace constraint
    trW = sum(tr(SumLij) for SumLij ∈ SumLijs)
    constraints += [real(trW) == 2^N]
    
    # Process matrix constraint
    IDX = CartesianIndices(Tuple(fill(0:1, N)))[2:2^N]
    for k in IDX
        Sk = [SumLij for SumLij in SumLijs]
        Sk = [Qpartialtrace(X, k) for X in Sk]
        Sk = reshape(Sk, fill(2, N)...)
        Hk = sum((-1)^(dot(Tuple(k), Tuple(j))) * Sk[j] for j in CartesianIndices(Tuple(fill(2, N))))
        constraints += [Hk == zeros(2^sum(Tuple(k)), 2^sum(Tuple(k)))]
    end

    # objective function
    linear = LinearIndices(Lijs)
    t = 1/2 * real(
        sum(
            Lijs[CartesianIndex(reverse(Tuple(a)))][linear[a], linear[a]] for a in CartesianIndices(Lijs)
        )
        )

    # SDP optimization
    problem = maximize(t, constraints)
    solve!(problem, Convex.MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs" => 1e-8, "eps_rel" => 1e-8))

    return (string(problem.status), problem.optval)
end

calc_causal(4)

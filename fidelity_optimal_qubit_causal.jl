using LinearAlgebra, MKL
using Convex, SCS
using Combinatorics
using QuantumInformation
using SparseArrays

function projector_s_s(N=1, atol=1e-3)
    
    """
        Function creating the projector onto commutator subspace.
        Usage: Take matrix X. Define |Y>> := projector_s_s(N) |X>>.
        Then, [Y,U^{⊗N+1}] = 0 for all U.
    """

    P = zeros(2^(2N+2), 2^(2N+2))
    idx = CartesianIndices(Tuple(fill(2, N+1)))

    for perm in permutations(1:N+1)
        x = zeros(fill(2, 2N+2)...)
        for k1 in idx
            k0 = CartesianIndex(Tuple(k1)[perm])
            x[k0, k1] = 1
        end
        x = vec(x)
        P += x * x'
    end

    F = eigen(P)
    Λ = Diagonal([x > atol ? 1 : 0 for x in F.values])
    P0 = F.vectors * Λ * F.vectors'
    sparse(P0)
end

function Qpartialtrace(Q, tuple)

    """
        Function defining partial trace of Q over qubit spaces X_j such that tuple[j] = 0.
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
        Function calculating the maximal value of the fidelity function for 
        d = 2 and N copies of given von Neumann measurement.  
    """

    P = projector_s_s(N)

    Rs = [ComplexVariable(2^(N+1), 2^(N+1)) for _=1:2^(N+1)]
    constraints = [R in :SDP for R in Rs]
    constraints += [P * vec(R) == vec(R) for R in Rs]

    Rs = reshape(Rs, fill(2, N+1)...)

    S = [sum(Rs[j, i] for i=1:2) for j ∈ CartesianIndices(Tuple(fill(2, N)))]
    S = reshape(S, fill(2, N)...)
    constraints += [S[j] == partialtrace(S[j], N+1, fill(2, N+1)) ⊗ I(2)/2 for j=1:2^N]
    S = [partialtrace(X, N+1, fill(2, N+1))/2 for X ∈ S]
    S = reshape(S, fill(2, N)...)
    trW = sum(tr(X) for X in S)
    constraints += [real(trW) == 2^N]
    
    IDX = CartesianIndices(Tuple(fill(0:1, N)))[2:2^N]
    for i in IDX
        Si = [X for X in S]
        Si = [Qpartialtrace(X, i) for X in Si]
        Si = reshape(Si, fill(2, N)...)
        Hi = sum((-1)^(dot(Tuple(i), Tuple(j))) * Si[j] for j in CartesianIndices(Tuple(fill(2, N))))
        constraints += [Hi == zeros(2^sum(Tuple(i)), 2^sum(Tuple(i)))]
    end

    linear = LinearIndices(Rs)
    t = 1/2 * real(
        sum(
            Rs[CartesianIndex(reverse(Tuple(a)))][linear[a], linear[a]] for a in CartesianIndices(Rs)
        )
        )
    problem = maximize(t, constraints)
    solve!(problem, Convex.MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs" => 1e-8, "eps_rel" => 1e-8))

    return (string(problem.status), problem.optval)
end

calc_causal(4)
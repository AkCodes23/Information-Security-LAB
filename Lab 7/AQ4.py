import time
import random
import math
from statistics import mean

# -------------------------------------------------
# Paillier (additively homomorphic)
# -------------------------------------------------

def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)

def L(u, n):
    return (u - 1) // n

def paillier_keygen(p, q):
    n = p * q
    lam = lcm(p-1, q-1)
    g = n + 1
    mu = pow(L(pow(g, lam, n*n), n), -1, n)
    return (n, g), (lam, mu)

def paillier_encrypt(pub, m):
    n, g = pub
    n2 = n * n
    r = random.randrange(1, n)
    while math.gcd(r, n) != 1:
        r = random.randrange(1, n)
    return (pow(g, m, n2) * pow(r, n, n2)) % n2

def paillier_decrypt(priv, pub, c):
    n, g = pub
    lam, mu = priv
    n2 = n * n
    u = pow(c, lam, n2)
    return (L(u, n) * mu) % n

def paillier_add_accumulate(cts, n):
    n2 = n*n
    acc = 1
    for c in cts:
        acc = (acc * c) % n2
    return acc

def bench_paillier(p, q, N=200):
    # Keygen
    t0 = time.perf_counter()
    pub, priv = paillier_keygen(p, q)
    t1 = time.perf_counter()

    # Encrypt N messages (uniform in [0, n-1])
    n, _ = pub
    msgs = [random.randrange(0, n) for _ in range(N)]
    t2 = time.perf_counter()
    cts = [paillier_encrypt(pub, m) for m in msgs]
    t3 = time.perf_counter()

    # Homomorphic sum (single aggregate)
    c_sum = paillier_add_accumulate(cts, n)
    t4 = time.perf_counter()

    # Decrypt aggregate
    _ = paillier_decrypt(priv, pub, c_sum)
    t5 = time.perf_counter()

    return {
        "keygen_s": t1 - t0,
        "encrypt_s": t3 - t2,
        "hom_op_s": t4 - t3,
        "decrypt_s": t5 - t4,
        "N": N
    }

# -------------------------------------------------
# ElGamal over Z_p* (multiplicatively homomorphic)
# -------------------------------------------------

def eg_modinv(a, p):
    return pow(a, p-2, p)

def elgamal_keygen(p=467, g=2):
    x = random.randrange(2, p - 2)  # secret
    y = pow(g, x, p)                # public component
    return (p, g, y), x

def elgamal_encrypt(pub, m):
    p, g, y = pub
    assert 1 <= m < p, "Plaintext must be in [1, p-1]"
    k = random.randrange(2, p - 2)
    a = pow(g, k, p)
    b = (pow(y, k, p) * m) % p
    return (a, b)

def elgamal_decrypt(priv, pub, ct):
    x = priv
    p, g, y = pub
    a, b = ct
    s = pow(a, x, p)
    return (b * eg_modinv(s, p)) % p

def elgamal_mul_accumulate(cts, p):
    a_acc, b_acc = 1, 1
    for a, b in cts:
        a_acc = (a_acc * a) % p
        b_acc = (b_acc * b) % p
    return (a_acc, b_acc)

def bench_elgamal(p=467, g=2, N=200):
    # Keygen
    t0 = time.perf_counter()
    pub, priv = elgamal_keygen(p, g)
    t1 = time.perf_counter()

    # Encrypt N messages (uniform in [1, p-1])
    P, _, _ = pub
    msgs = [random.randrange(1, P) for _ in range(N)]
    t2 = time.perf_counter()
    cts = [elgamal_encrypt(pub, m) for m in msgs]
    t3 = time.perf_counter()

    # Homomorphic product (single aggregate)
    c_prod = elgamal_mul_accumulate(cts, P)
    t4 = time.perf_counter()

    # Decrypt aggregate
    _ = elgamal_decrypt(priv, pub, c_prod)
    t5 = time.perf_counter()

    return {
        "keygen_s": t1 - t0,
        "encrypt_s": t3 - t2,
        "hom_op_s": t4 - t3,
        "decrypt_s": t5 - t4,
        "N": N
    }

# -------------------------------------------------
# Driver: multi-trial benchmarking and report
# -------------------------------------------------

def summarize(name, res_list):
    N = res_list[0]["N"]
    avg_keygen = mean(r["keygen_s"] for r in res_list)
    avg_encrypt = mean(r["encrypt_s"] for r in res_list)
    avg_hom = mean(r["hom_op_s"] for r in res_list)
    avg_dec = mean(r["decrypt_s"] for r in res_list)
    enc_throughput = N / avg_encrypt if avg_encrypt > 0 else float("inf")
    return {
        "scheme": name,
        "N": N,
        "keygen_s": avg_keygen,
        "encrypt_s": avg_encrypt,
        "enc_msgs_per_s": enc_throughput,
        "hom_op_s": avg_hom,
        "decrypt_s": avg_dec
    }

def main():
    random.seed(42)

    # Parameters (tune these)
    TRIALS = 5
    N = 500

    # Paillier primes (demo sizes; increase for heavier runs)
    p_pail, q_pail = 2357, 2551

    # ElGamal field (demo prime and generator)
    P_eg, G_eg = 467, 2

    # Warm-up to avoid first-run overhead noise
    bench_paillier(p_pail, q_pail, N=50)
    bench_elgamal(P_eg, G_eg, N=50)

    # Run trials
    paillier_results = [bench_paillier(p_pail, q_pail, N=N) for _ in range(TRIALS)]
    elgamal_results = [bench_elgamal(P_eg, G_eg, N=N) for _ in range(TRIALS)]

    s_p = summarize("Paillier (add)", paillier_results)
    s_e = summarize("ElGamal (mul)", elgamal_results)

    # Pretty print summary
    def fmt(x, digits=6):
        return f"{x:.{digits}f}" if isinstance(x, float) else str(x)

    print("\n=== Benchmark Summary (averaged) ===")
    headers = ["Scheme", "N", "KeyGen(s)", "Encrypt(s)", "Enc msgs/s", "HomOp(s)", "Decrypt(s)"]
    row_p = [s_p["scheme"], s_p["N"], fmt(s_p["keygen_s"]), fmt(s_p["encrypt_s"]),
             fmt(s_p["enc_msgs_per_s"]), fmt(s_p["hom_op_s"]), fmt(s_p["decrypt_s"])]
    row_e = [s_e["scheme"], s_e["N"], fmt(s_e["keygen_s"]), fmt(s_e["encrypt_s"]),
             fmt(s_e["enc_msgs_per_s"]), fmt(s_e["hom_op_s"]), fmt(s_e["decrypt_s"])]

    # Simple table
    col_widths = [max(len(h), len(str(rp))) for h, rp in zip(headers, row_p)]
    col_widths = [max(w, len(str(re))) for w, re in zip(col_widths, row_e)]

    def print_row(cells):
        print("  ".join(str(c).ljust(w) for c, w in zip(cells, col_widths)))

    print_row(headers)
    print_row(row_p)
    print_row(row_e)

if __name__ == "__main__":
    main()


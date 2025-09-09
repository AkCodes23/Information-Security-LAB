# Lab 4D: RSA vulnerable key attack via Fermat factorization (close primes)
# Requires: pycryptodome
# Run: python lab4_rsa_weak_attack.py

from math import isqrt
from Crypto.PublicKey import RSA
from Crypto.Util.number import getPrime, inverse

def fermat_factor(n):
    a = isqrt(n)
    if a*a < n: a += 1
    b2 = a*a - n
    while not isqrt(b2)**2 == b2:
        a += 1
        b2 = a*a - n
    b = isqrt(b2)
    p = a - b
    q = a + b
    return p, q

def gen_weak_rsa(bits=1024, gap=1<<100):
    # primes p,q close to each other (unsafe)
    p = getPrime(bits//2)
    q = p + gap  # not guaranteed prime, simple demo: find next prime-ish
    while True:
        # crude increment until prime
        from Crypto.Util.number import isPrime
        if isPrime(q): break
        q += 1
    n = p*q
    e = 65537
    phi = (p-1)*(q-1)
    d = inverse(e, phi)
    return n,e,d,p,q

if __name__ == "__main__":
    n,e,d,p,q = gen_weak_rsa(512, gap=1<<50)
    print("Attacking n...")
    fp,fq = fermat_factor(n)
    print("Recovered?", sorted([p,q]) == sorted([fp,fq]))
    # Mitigation: use strong random primes far apart; use proven libraries; validate key generation.

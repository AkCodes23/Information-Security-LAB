# benchmark_suite.py
# Benchmarks symmetric (DES/AES modes), asymmetric (RSA/ECIES), hashing, and PHE (Paillier/ElGamal).
# Saves CSVs and PNG plots under --output directory.
# Usage examples:
#   python benchmark_suite.py --all
#   python benchmark_suite.py --symmetric --sizes 1_000 100_000 1_000_000
#   python benchmark_suite.py --asymmetric --rsa_bits 2048 --sizes 1_000_000 10_000_000
#   python benchmark_suite.py --hash --strings 10000
#   python benchmark_suite.py --phe --phe_N 2000

import os
import csv
import time
import math
import argparse
import secrets
import random
import string
from statistics import mean
import matplotlib.pyplot as plt

# Crypto deps
from Crypto.Cipher import DES, DES3, AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

# ------------- Utility I/O and plotting -------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def write_csv(path, headers, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)

def plot_bar(path, title, labels, values, ylabel="seconds"):
    plt.figure(figsize=(10,5))
    plt.bar(labels, values, color="#3b82f6")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_lines(path, title, x_values, series_dict, xlabel="bytes", ylabel="seconds"):
    plt.figure(figsize=(10,5))
    for name, vals in series_dict.items():
        plt.plot(x_values, vals, marker="o", label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# ------------- Symmetric benchmarks -------------

def sym_encrypt_decrypt(cipher_ctor, block_size, data: bytes):
    # Pad for block modes; CTR handles streaming (no pad) if supported
    c = cipher_ctor()
    needs_pad = getattr(c, "_needs_pad", True)
    if needs_pad:
        ct = c.encrypt(pad(data, block_size))
        pt = unpad(cipher_ctor().decrypt(ct), block_size)
    else:
        ct = c.encrypt(data)
        pt = cipher_ctor().decrypt(ct)
    return ct, pt

def bench_symmetric(output_dir, sizes=(1_000, 100_000, 1_000_000), trials=3):
    # Keys/IVs
    des_key = b"A1B2C3D4"[:8]
    des3_key = bytes.fromhex("1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF")
    aes128 = get_random_bytes(16)
    aes192 = get_random_bytes(24)
    aes256 = get_random_bytes(32)
    iv8 = b"12345678"
    iv16 = get_random_bytes(16)
    nonce8 = b"\x00"*8  # CTR

    # Cipher constructors with attribute for padding need
    def DES_ECB():
        c = DES.new(des_key, DES.MODE_ECB)
        c._needs_pad = True
        return c
    def DES_CBC():
        c = DES.new(des_key, DES.MODE_CBC, iv=iv8)
        c._needs_pad = True
        return c
    def DES3_ECB():
        c = DES3.new(des3_key, DES3.MODE_ECB)
        c._needs_pad = True
        return c
    def AES128_ECB():
        c = AES.new(aes128, AES.MODE_ECB)
        c._needs_pad = True
        return c
    def AES256_ECB():
        c = AES.new(aes256, AES.MODE_ECB)
        c._needs_pad = True
        return c
    def AES128_CBC():
        c = AES.new(aes128, AES.MODE_CBC, iv=iv16)
        c._needs_pad = True
        return c
    def AES256_CBC():
        c = AES.new(aes256, AES.MODE_CBC, iv=iv16)
        c._needs_pad = True
        return c
    def AES128_CTR():
        c = AES.new(aes128, AES.MODE_CTR, nonce=nonce8)
        c._needs_pad = False
        return c
    def AES256_CTR():
        c = AES.new(aes256, AES.MODE_CTR, nonce=nonce8)
        c._needs_pad = False
        return c

    suites = [
        ("DES-ECB", DES_ECB, 8),
        ("DES-CBC", DES_CBC, 8),
        ("3DES-ECB", DES3_ECB, 8),
        ("AES128-ECB", AES128_ECB, 16),
        ("AES256-ECB", AES256_ECB, 16),
        ("AES128-CBC", AES128_CBC, 16),
        ("AES256-CBC", AES256_CBC, 16),
        ("AES128-CTR", AES128_CTR, 16),
        ("AES256-CTR", AES256_CTR, 16),
    ]

    rows = [("suite","size_bytes","enc_s","dec_s","throughput_MBps")]
    plot_series = {name: [] for name,_,_ in suites}
    for size in sizes:
        data = os.urandom(size)
        for name, ctor, block_sz in suites:
            enc_times, dec_times = [], []
            for _ in range(trials):
                t0 = time.perf_counter()
                c = ctor()
                needs_pad = getattr(c, "_needs_pad", True)
                if needs_pad:
                    ct = c.encrypt(pad(data, block_sz))
                    t1 = time.perf_counter()
                    pt = ctor().decrypt(ct)
                    _ = unpad(pt, block_sz)
                    t2 = time.perf_counter()
                else:
                    ct = c.encrypt(data); t1=time.perf_counter()
                    _ = ctor().decrypt(ct); t2=time.perf_counter()
                enc_times.append(t1-t0); dec_times.append(t2-t1)
            enc_s = mean(enc_times); dec_s = mean(dec_times)
            thr = (size / (1024*1024)) / enc_s if enc_s > 0 else float("inf")
            rows.append((name, size, enc_s, dec_s, thr))
            plot_series[name].append(enc_s)
            print(f"[SYM] {name} size={size}: enc={enc_s:.6f}s dec={dec_s:.6f}s thr={thr:.2f}MB/s")

    csv_path = os.path.join(output_dir, "symmetric_times.csv")
    write_csv(csv_path, rows[0], rows[1:])
    plot_lines(os.path.join(output_dir, "symmetric_enc_times.png"),
               "Symmetric encryption time vs size", list(sizes), plot_series)

# ------------- Asymmetric benchmarks (RSA, ECIES hybrid for files) -------------

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

def ec_keypair():
    priv = ec.generate_private_key(ec.SECP256R1())
    pub = priv.public_key()
    return priv, pub

def ecies_encrypt(pub, plaintext: bytes):
    eph_priv = ec.generate_private_key(ec.SECP256R1())
    shared = eph_priv.exchange(ec.ECDH(), pub)
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"ecies").derive(shared)
    iv = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
    ct, tag = cipher.encrypt_and_digest(plaintext)
    eph_pub_bytes = eph_priv.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return eph_pub_bytes, iv, tag, ct

def ecies_decrypt(priv, package):
    eph_pub_bytes, iv, tag, ct = package
    eph_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), eph_pub_bytes)
    shared = priv.exchange(ec.ECDH(), eph_pub)
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"ecies").derive(shared)
    cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
    pt = cipher.decrypt_and_verify(ct, tag)
    return pt

def rsa_file_encrypt(pub, data: bytes):
    sym = get_random_bytes(32)
    iv = get_random_bytes(12)
    c = AES.new(sym, AES.MODE_GCM, nonce=iv)
    cdata, tag = c.encrypt_and_digest(data)
    wrapped = PKCS1_OAEP.new(pub).encrypt(sym)
    return wrapped, iv, tag, cdata

def rsa_file_decrypt(priv, package):
    wrapped, iv, tag, cdata = package
    sym = PKCS1_OAEP.new(priv).decrypt(wrapped)
    pt = AES.new(sym, AES.MODE_GCM, nonce=iv).decrypt_and_verify(cdata, tag)
    return pt

def bench_asymmetric(output_dir, sizes=(1_000_000, 10_000_000), rsa_bits=2048, trials=3):
    # Keygen timing
    t0 = time.perf_counter()
    rsa_priv = RSA.generate(rsa_bits)
    rsa_pub = rsa_priv.publickey()
    t1 = time.perf_counter()
    ecc_priv, ecc_pub = ec_keypair()
    t2 = time.perf_counter()
    print(f"[ASYM] RSA-{rsa_bits} keygen: {t1-t0:.4f}s | EC P-256 keygen: {t2-t1:.4f}s")

    rows = [("scheme","size_bytes","enc_s","dec_s","throughput_MBps")]
    series_enc = {"RSA-{}+AES-GCM".format(rsa_bits): [], "ECIES(P-256)+AES-GCM": []}

    for size in sizes:
        data = os.urandom(size)
        # RSA+AES
        enc_times, dec_times = [], []
        for _ in range(trials):
            t0 = time.perf_counter(); pkg = rsa_file_encrypt(rsa_pub, data); t1 = time.perf_counter()
            _ = rsa_file_decrypt(rsa_priv, pkg); t2 = time.perf_counter()
            enc_times.append(t1-t0); dec_times.append(t2-t1)
        e, d = mean(enc_times), mean(dec_times)
        thr = (size/(1024*1024))/e
        rows.append(("RSA{}+AESGCM".format(rsa_bits), size, e, d, thr))
        series_enc["RSA-{}+AES-GCM".format(rsa_bits)].append(e)
        print(f"[ASYM] RSA{rsa_bits}+AES size={size}: enc={e:.4f}s dec={d:.4f}s thr={thr:.2f}MB/s")

        # ECIES
        enc_times, dec_times = [], []
        for _ in range(trials):
            t0 = time.perf_counter(); pkg = ecies_encrypt(ecc_pub, data); t1 = time.perf_counter()
            _ = ecies_decrypt(ecc_priv, pkg); t2 = time.perf_counter()
            enc_times.append(t1-t0); dec_times.append(t2-t1)
        e, d = mean(enc_times), mean(dec_times)
        thr = (size/(1024*1024))/e
        rows.append(("ECIES+AESCGM", size, e, d, thr))
        series_enc["ECIES(P-256)+AES-GCM"].append(e)
        print(f"[ASYM] ECIES size={size}: enc={e:.4f}s dec={d:.4f}s thr={thr:.2f}MB/s")

    csv_path = os.path.join(output_dir, "asymmetric_times.csv")
    write_csv(csv_path, rows[0], rows[1:])
    plot_lines(os.path.join(output_dir, "asymmetric_enc_times.png"),
               "Hybrid asymmetric encryption time vs size", list(sizes), series_enc)

# ------------- Hashing benchmarks -------------

import hashlib

def bench_hash(output_dir, strings=5000, maxlen=128):
    msgs = ["".join(random.choice(string.ascii_letters) for _ in range(random.randint(8, maxlen))).encode() for _ in range(strings)]
    algs = [("MD5", hashlib.md5), ("SHA1", hashlib.sha1), ("SHA256", hashlib.sha256)]
    rows = [("algorithm","strings","total_time_s","throughput_hashes_per_s","collisions")]
    labels, values = [], []
    for name, fn in algs:
        t0 = time.perf_counter()
        digests = [fn(m).hexdigest() for m in msgs]
        t1 = time.perf_counter()
        total = t1 - t0
        thr = strings / total if total > 0 else float("inf")
        # collision check (unlikely)
        seen = set(); collisions = 0
        for d in digests:
            if d in seen: collisions += 1
            seen.add(d)
        rows.append((name, strings, total, thr, collisions))
        labels.append(name); values.append(total)
        print(f"[HASH] {name}: total={total:.6f}s thr={thr:.0f}/s collisions={collisions}")

    csv_path = os.path.join(output_dir, "hash_times.csv")
    write_csv(csv_path, rows[0], rows[1:])
    plot_bar(os.path.join(output_dir, "hash_times.png"),
             f"Hash total time for {strings} strings", labels, values, ylabel="seconds")

# ------------- PHE benchmarks (Paillier, ElGamal) -------------

def lcm(a,b): return abs(a*b)//math.gcd(a,b)
def L(u,n): return (u-1)//n

def paillier_keygen(p,q):
    n = p*q; lam = lcm(p-1,q-1); g = n+1
    mu = pow(L(pow(g, lam, n*n), n), -1, n)
    return (n,g), (lam,mu)

def paillier_encrypt(pub, m):
    n,g = pub; n2 = n*n
    r = random.randrange(1,n)
    while math.gcd(r,n) != 1: r = random.randrange(1,n)
    return (pow(g,m,n2) * pow(r,n,n2)) % n2

def paillier_decrypt(priv, pub, c):
    n,g = pub; lam,mu = priv; n2 = n*n
    u = pow(c, lam, n2)
    return (L(u,n) * mu) % n

def paillier_add(c1,c2,n): return (c1*c2) % (n*n)

def elgamal_keygen(p=467, g=2):
    x = random.randrange(2,p-2)
    y = pow(g,x,p)
    return (p,g,y), x

def elgamal_encrypt(pub, m):
    p,g,y = pub
    k = random.randrange(2,p-2)
    a = pow(g,k,p); b = (pow(y,k,p)*m) % p
    return (a,b)

def elgamal_decrypt(priv, pub, ct):
    x = priv; p,g,y = pub; a,b = ct
    s = pow(a,x,p)
    return (b * pow(s, p-2, p)) % p

def bench_phe(output_dir, phe_N=2000, trials=3):
    # Paillier demo primes (small for speed)
    p_p, q_p = 1789, 2027
    pub_p, priv_p = paillier_keygen(p_p, q_p)
    n, _ = pub_p

    # ElGamal small field
    pub_e, priv_e = elgamal_keygen()

    # Messages for PHE
    msgs = [random.randrange(0, min(n, pub_e[0])) for _ in range(phe_N)]

    # Paillier
    t0 = time.perf_counter()
    cts = [paillier_encrypt(pub_p, m) for m in msgs]
    t1 = time.perf_counter()
    acc = 1
    n2 = n*n
    for c in cts:
        acc = (acc * c) % n2
    t2 = time.perf_counter()
    _ = paillier_decrypt(priv_p, pub_p, acc)
    t3 = time.perf_counter()
    print(f"[PHE] Paillier N={phe_N}: enc={t1-t0:.4f}s hom_add={t2-t1:.4f}s dec={t3-t2:.4f}s")

    # ElGamal (multiplicative)
    t0 = time.perf_counter()
    cts_e = [elgamal_encrypt(pub_e, max(1, m % (pub_e[0]-1))) for m in msgs]  # in [1,p-1]
    t1 = time.perf_counter()
    a_acc, b_acc = 1, 1
    p = pub_e[0]
    for a,b in cts_e:
        a_acc = (a_acc * a) % p; b_acc = (b_acc * b) % p
    t2 = time.perf_counter()
    _ = elgamal_decrypt(priv_e, pub_e, (a_acc, b_acc))
    t3 = time.perf_counter()
    print(f"[PHE] ElGamal N={phe_N}: enc={t1-t0:.4f}s hom_mul={t2-t1:.4f}s dec={t3-t2:.4f}s")

    rows = [
        ("scheme","N","enc_s","hom_op_s","dec_s"),
        ("Paillier", phe_N, (t1-t0), (t2-t1), (t3-t2)),
        ("ElGamal", phe_N, (t1-t0), (t2-t1), (t3-t2)),
    ]
    csv_path = os.path.join(output_dir, "phe_times.csv")
    write_csv(csv_path, rows[0], rows[1:])
    plot_bar(os.path.join(output_dir, "phe_enc.png"),
             "PHE encryption time (N messages)", ["Paillier","ElGamal"], [rows[1][2], rows[2][2]])
    plot_bar(os.path.join(output_dir, "phe_hom.png"),
             "PHE homomorphic aggregate time", ["Paillier","ElGamal"], [rows[1][3], rows[2][3]])
    plot_bar(os.path.join(output_dir, "phe_dec.png"),
             "PHE decryption time", ["Paillier","ElGamal"], [rows[1][4], rows[2][4]])

# ------------- CLI -------------

def parse_args():
    p = argparse.ArgumentParser(description="Crypto benchmarking suite with plots")
    p.add_argument("--output", default="bench_out", help="Output directory")
    p.add_argument("--all", action="store_true", help="Run all suites")
    p.add_argument("--symmetric", action="store_true", help="Run symmetric benchmarks")
    p.add_argument("--asymmetric", action="store_true", help="Run asymmetric benchmarks")
    p.add_argument("--hash", action="store_true", help="Run hash benchmarks")
    p.add_argument("--phe", action="store_true", help="Run PHE benchmarks")
    p.add_argument("--sizes", nargs="*", type=lambda x: int(x.replace("_","")), default=None,
                   help="Message/file sizes for sym/asym (bytes)")
    p.add_argument("--rsa_bits", type=int, default=2048, help="RSA key size for asymmetric suite")
    p.add_argument("--strings", type=int, default=5000, help="Number of random strings for hash suite")
    p.add_argument("--phe_N", type=int, default=2000, help="Number of messages for PHE suite")
    return p.parse_args()

def main():
    args = parse_args()
    out = ensure_dir(args.output)
    run_any = args.all or args.symmetric or args.asymmetric or args.hash or args.phe
    if not run_any:
        print("No suite selected. Use --all or one of --symmetric --asymmetric --hash --phe.")
        return
    sizes = args.sizes if args.sizes else None

    if args.all or args.symmetric:
        sym_sizes = sizes if sizes else (1_000, 100_000, 1_000_000)
        print("\n=== Symmetric benchmarks ===")
        bench_symmetric(out, sizes=sym_sizes)

    if args.all or args.asymmetric:
        asym_sizes = sizes if sizes else (1_000_000, 10_000_000)
        print("\n=== Asymmetric (hybrid) benchmarks ===")
        bench_asymmetric(out, sizes=asym_sizes, rsa_bits=args.rsa_bits)

    if args.all or args.hash:
        print("\n=== Hash benchmarks ===")
        bench_hash(out, strings=args.strings)

    if args.all or args.phe:
        print("\n=== PHE benchmarks ===")
        bench_phe(out, phe_N=args.phe_N)

    print(f"\nDone. CSVs and PNGs saved in: {os.path.abspath(out)}")

if __name__ == "__main__":
    main()

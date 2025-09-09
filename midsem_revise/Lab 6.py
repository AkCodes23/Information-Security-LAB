# Lab 6: ElGamal, Schnorr signatures (mod p), Diffie–Hellman, client-server, CIA with RSA+SHA
# Requires: pycryptodome
# Run: python lab6_adv_asymmetric.py

import socket, threading, time, hashlib, secrets
from Crypto.PublicKey import RSA, ElGamal
from Crypto.Cipher import PKCS1_OAEP, AES
from Crypto.Random import get_random_bytes, random
from Crypto.Util.number import getPrime, inverse

# ElGamal (message int)
def elgamal_demo():
    key = ElGamal.generate(2048, get_random_bytes)
    m = int.from_bytes(b"ElGamal Demo", "big") % key.p
    k = random.StrongRandom().randint(1, key.p-2)
    ct = key.publickey().encrypt(m, k)
    dec = key.decrypt(ct)
    return dec.to_bytes((dec.bit_length()+7)//8, "big")

# Schnorr signature (mod p, q|p-1)
def schnorr_params(bits=256):
    # Generate p= kq+1 with q large prime
    q = getPrime(bits//2)
    k = 2
    while True:
        p = k*q + 1
        if pow(2, p-1, p) == 1 and p.bit_length() >= bits:
            break
        k += 1
    g = pow(2, k, p)
    return p,q,g

def schnorr_keygen(p,q,g):
    x = secrets.randbelow(q-1)+1
    y = pow(g, x, p)
    return x,y

def schnorr_sign(p,q,g,x, msg: bytes):
    r = secrets.randbelow(q-1)+1
    a = pow(g, r, p)
    e = int.from_bytes(hashlib.sha256(a.to_bytes(256,'big')+msg).digest(), 'big') % q
    s = (r + x*e) % q
    return (e, s)

def schnorr_verify(p,q,g,y,msg, sig):
    e,s = sig
    a = (pow(g, s, p) * pow(y, -e, p)) % p
    e2 = int.from_bytes(hashlib.sha256(a.to_bytes(256,'big')+msg).digest(), 'big') % q
    return e == e2

# Diffie–Hellman
def dh_keypair(bits=512):
    p = getPrime(bits); g = 2
    a = secrets.randbelow(p-2)+2
    A = pow(g,a,p)
    return p,g,a,A

def dh_shared(p,g,a,B):
    return pow(B,a,p)

# Client-server RSA+AES channel (CIA demo)
def server_rsa(host="127.0.0.1", port=6001):
    srv = socket.socket(); srv.bind((host, port)); srv.listen(1)
    conn, _ = srv.accept()
    # receive client's RSA public key (PEM)
    pem_len = int.from_bytes(conn.recv(4), 'big')
    client_pub_pem = conn.recv(pem_len)
    client_pub = RSA.import_key(client_pub_pem)
    # send server RSA public key
    srv_priv = RSA.generate(2048)
    conn.sendall(len(srv_priv.publickey().export_key()).to_bytes(4,'big') + srv_priv.publickey().export_key())
    # receive encrypted message + signature digest
    enc_len = int.from_bytes(conn.recv(4),'big'); enc = conn.recv(enc_len)
    sig_len = int.from_bytes(conn.recv(4),'big'); sig = conn.recv(sig_len)
    # decrypt
    msg = PKCS1_OAEP.new(srv_priv).decrypt(enc)
    # verify integrity via SHA256 and signature (simulate: client includes hash)
    ok = hashlib.sha256(msg).digest() == sig
    # respond with AES-protected ack
    sym = get_random_bytes(32)
    wrapped = PKCS1_OAEP.new(client_pub).encrypt(sym)
    ct = AES.new(sym, AES.MODE_GCM)
    ack, tag = ct.encrypt_and_digest(b"ACK:" + (b"OK" if ok else b"FAIL"))
    conn.sendall(len(wrapped).to_bytes(4,'big') + wrapped + ct.nonce + tag + ack)
    conn.close(); srv.close()

def client_rsa(host="127.0.0.1", port=6001):
    c = socket.socket(); c.connect((host, port))
    priv = RSA.generate(2048); pub_pem = priv.publickey().export_key()
    c.sendall(len(pub_pem).to_bytes(4,'big') + pub_pem)
    # recv server pub
    slen = int.from_bytes(c.recv(4),'big'); srv_pub = RSA.import_key(c.recv(slen))
    # send encrypted message and its hash (confidentiality + integrity)
    msg = b"Confidentiality Integrity Availability"
    enc = PKCS1_OAEP.new(srv_pub).encrypt(msg)
    sig = hashlib.sha256(msg).digest()
    c.sendall(len(enc).to_bytes(4,'big') + enc + len(sig).to_bytes(4,'big') + sig)
    # receive AES-wrapped ack
    wlen = int.from_bytes(c.recv(4),'big'); wrapped = c.recv(wlen)
    nonce = c.recv(12); tag = c.recv(16); ack = c.recv(1024)
    sym = PKCS1_OAEP.new(priv).decrypt(wrapped)
    resp = AES.new(sym, AES.MODE_GCM, nonce=nonce).decrypt_and_verify(ack, tag)
    print("Server response:", resp)
    c.close()

if __name__ == "__main__":
    # ElGamal demo
    print("ElGamal dec:", elgamal_demo())

    # Schnorr demo
    p,q,g = schnorr_params(256)
    x,y = schnorr_keygen(p,q,g)
    msg = b"Schnorr Signature"
    sig = schnorr_sign(p,q,g,x,msg)
    print("Schnorr verify:", schnorr_verify(p,q,g,y,msg,sig))

    # DH
    p,g,a,A = dh_keypair()
    _,_,b,B = dh_keypair()
    print("DH equal shared:", dh_shared(p,g,a,B) == dh_shared(p,g,b,A))

    # Client-server CIA demo
    t = threading.Thread(target=server_rsa, daemon=True); t.start()
    time.sleep(0.2)
    client_rsa()

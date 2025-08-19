# pip install pycryptodome
import json, time, secrets, hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
from Crypto.Cipher import AES
from Crypto.Util import number

# ---------------- Key Manager ----------------

@dataclass
class RSAIdentity:
    name: str
    public_pem: bytes
    private_pem: bytes
    revoked: bool = False

class KeyManager:
    def __init__(self):
        self._ids: Dict[str, RSAIdentity] = {}

    def register(self, name: str, bits: int = 2048) -> RSAIdentity:
        key = RSA.generate(bits)
        ident = RSAIdentity(
            name=name,
            public_pem=key.publickey().export_key(),
            private_pem=key.export_key(),
        )
        self._ids[name] = ident
        return ident

    def get_public(self, name: str) -> Optional[bytes]:
        ident = self._ids.get(name)
        if ident and not ident.revoked:
            return ident.public_pem
        return None

    def get_private(self, name: str) -> Optional[bytes]:
        ident = self._ids.get(name)
        if ident and not ident.revoked:
            return ident.private_pem
        return None

    def revoke(self, name: str) -> bool:
        ident = self._ids.get(name)
        if not ident:
            return False
        ident.revoked = True
        return True

# ---------------- Crypto helpers ----------------

def dh_generate_parameters(bits: int = 2048) -> Tuple[int, int]:
    # Generate a strong prime p and use generator g=2
    p = number.getStrongPrime(bits)
    g = 2
    return p, g

def dh_keypair(p: int, g: int, bits: int = 256) -> Tuple[int, int]:
    # Ephemeral secret a and public A = g^a mod p
    a = secrets.randbits(bits)
    A = pow(g, a, p)
    return a, A

def dh_shared_secret(p: int, peer_pub: int, my_priv: int) -> bytes:
    s = pow(peer_pub, my_priv, p)
    # Derive a 256-bit key using SHA-256(s || "context")
    h = hashlib.sha256(int.to_bytes(s, (s.bit_length() + 7) // 8, 'big') + b"SecureCorp-DH").digest()
    return h  # 32 bytes

def aes_gcm_encrypt(key: bytes, plaintext: bytes, aad: bytes = b"") -> Tuple[bytes, bytes, bytes]:
    nonce = secrets.token_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    cipher.update(aad)
    ct, tag = cipher.encrypt_and_digest(plaintext)
    return nonce, ct, tag

def aes_gcm_decrypt(key: bytes, nonce: bytes, ct: bytes, tag: bytes, aad: bytes = b"") -> bytes:
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    cipher.update(aad)
    return cipher.decrypt_and_verify(ct, tag)

def rsa_sign(private_pem: bytes, data: bytes) -> bytes:
    key = RSA.import_key(private_pem)
    h = SHA256.new(data)
    return pkcs1_15.new(key).sign(h)

def rsa_verify(public_pem: bytes, data: bytes, sig: bytes) -> bool:
    key = RSA.import_key(public_pem)
    h = SHA256.new(data)
    try:
        pkcs1_15.new(key).verify(h, sig)
        return True
    except (ValueError, TypeError):
        return False

# ---------------- Subsystem node ----------------

class Subsystem:
    def __init__(self, name: str, km: KeyManager):
        self.name = name
        self.km = km
        if not km.get_private(name):
            km.register(name)

    def public_key(self) -> bytes:
        pk = self.km.get_public(self.name)
        if not pk:
            raise RuntimeError(f"{self.name} is revoked or unknown")
        return pk

    def private_key(self) -> bytes:
        sk = self.km.get_private(self.name)
        if not sk:
            raise RuntimeError(f"{self.name} is revoked or unknown")
        return sk

    def initiate_channel(self, peer_name: str, document: bytes) -> Dict:
        # 1) Setup ephemeral DH
        p, g = dh_generate_parameters(bits=2048)
        a, A = dh_keypair(p, g)
        # 2) Sign transcript
        transcript = json.dumps({"p": str(p), "g": g, "A": str(A), "from": self.name, "to": peer_name}).encode()
        sig = rsa_sign(self.private_key(), transcript)
        # 3) Build handshake message
        return {"handshake": {"p": str(p), "g": g, "A": str(A), "from": self.name, "to": peer_name, "sig": sig.hex()},
                "state": {"p": p, "g": g, "a": a, "A": A, "document": document}}

    def respond_and_encrypt(self, incoming: Dict, peer_pubkey: bytes) -> Dict:
        hs = incoming["handshake"]
        p = int(hs["p"]); g = hs["g"]; A = int(hs["A"])
        transcript = json.dumps({"p": hs["p"], "g": g, "A": hs["A"], "from": hs["from"], "to": hs["to"]}).encode()
        if not rsa_verify(peer_pubkey, transcript, bytes.fromhex(hs["sig"])):
            raise RuntimeError("Peer authentication failed")
        b, B = dh_keypair(p, g)
        key = dh_shared_secret(p, A, b)
        aad = f"{hs['from']}->{hs['to']}".encode()
        nonce, ct, tag = aes_gcm_encrypt(key, incoming["state"]["document"], aad=aad)
        my_transcript = json.dumps({"p": hs["p"], "g": g, "B": str(B), "from": hs["to"], "to": hs["from"]}).encode()
        my_sig = rsa_sign(self.private_key(), my_transcript)
        return {
            "cipher": {"nonce": nonce.hex(), "ct": ct.hex(), "tag": tag.hex(), "aad": aad.hex()},
            "reply": {"p": hs["p"], "g": g, "B": str(B), "from": hs["to"], "to": hs["from"], "sig": my_sig.hex()},
            "state": {"p": p, "b": b, "B": B},
        }

    def finalize_and_decrypt(self, response: Dict, peer_pubkey: bytes, state: Dict) -> bytes:
        rep = response["reply"]
        p = int(rep["p"]); B = int(rep["B"])
        transcript = json.dumps({"p": rep["p"], "g": rep["g"], "B": rep["B"], "from": rep["from"], "to": rep["to"]}).encode()
        if not rsa_verify(peer_pubkey, transcript, bytes.fromhex(rep["sig"])):
            raise RuntimeError("Peer authentication failed")
        key = dh_shared_secret(int(state["p"]), B, int(state["a"]))
        c = response["cipher"]
        plaintext = aes_gcm_decrypt(key,
                                    bytes.fromhex(c["nonce"]),
                                    bytes.fromhex(c["ct"]),
                                    bytes.fromhex(c["tag"]),
                                    aad=bytes.fromhex(c["aad"]))
        return plaintext

# ---------------- Demo ----------------

def demo():
    km = KeyManager()
    finance = Subsystem("SystemA_Finance", km)
    hr = Subsystem("SystemB_HR", km)
    scm = Subsystem("SystemC_SupplyChain", km)

    doc = b"Q3 consolidated financial report v1.2"
    # A -> B
    hs = finance.initiate_channel("SystemB_HR", doc)
    resp = hr.respond_and_encrypt(hs, km.get_public("SystemA_Finance"))
    clear = finance.finalize_and_decrypt(resp, km.get_public("SystemB_HR"), hs["state"])
    print("Decrypted @HR:", clear.decode())

    # Revoke HR, demonstrate failure
    km.revoke("SystemB_HR")
    try:
        hr.respond_and_encrypt(hs, km.get_public("SystemA_Finance"))
    except Exception as e:
        print("Expected failure (revoked HR):", str(e))

    # Add new subsystem later (scales cleanly)
    legal = Subsystem("SystemD_Legal", km)
    msg = b"Employee contract template v3"
    hs2 = legal.initiate_channel("SystemC_SupplyChain", msg)
    resp2 = scm.respond_and_encrypt(hs2, km.get_public("SystemD_Legal"))
    clear2 = legal.finalize_and_decrypt(resp2, km.get_public("SystemC_SupplyChain"), hs2["state"])
    print("Decrypted @Legal:", clear2.decode())

if __name__ == "__main__":
    demo()

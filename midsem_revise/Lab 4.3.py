# Lab 4C: DRM using ElGamal KMS + Access Control
# Requires: pycryptodome
# Run: python lab4_drm_elgamal.py

from Crypto.PublicKey import ElGamal
from Crypto.Random import get_random_bytes, random
from Crypto.Cipher import AES

class DRM:
    def __init__(self):
        self.master = ElGamal.generate(2048, get_random_bytes)
        self.permissions = {}  # user -> set(content_id)

    def encrypt_content(self, content_id: str, data: bytes):
        sym = get_random_bytes(32)
        ct = AES.new(sym, AES.MODE_GCM)
        c, tag = ct.encrypt_and_digest(data)
        # wrap sym via ElGamal: represent as int < p
        m_int = int.from_bytes(sym, "big") % self.master.p
        k = random.StrongRandom().randint(1, self.master.p-2)
        wrapped = self.master.publickey().encrypt(m_int, k)
        return {"content": c, "tag": tag, "nonce": ct.nonce, "wrap": wrapped}

    def decrypt_content(self, pkg):
        m_int = self.master.decrypt(pkg["wrap"])
        sym = m_int.to_bytes(32, "big")
        return AES.new(sym, AES.MODE_GCM, nonce=pkg["nonce"]).decrypt_and_verify(pkg["content"], pkg["tag"])

    def grant(self, user, content_id):
        self.permissions.setdefault(user, set()).add(content_id)

    def revoke(self, user, content_id):
        self.permissions.setdefault(user, set()).discard(content_id)

    def can_access(self, user, content_id):
        return content_id in self.permissions.get(user, set())

if __name__ == "__main__":
    drm = DRM()
    drm.grant("alice", "movie1")
    pkg = drm.encrypt_content("movie1", b"BinaryMovieData...")
    if drm.can_access("alice", "movie1"):
        print("Decrypted:", drm.decrypt_content(pkg))
    drm.revoke("alice", "movie1")
    print("Access after revoke:", drm.can_access("alice", "movie1"))

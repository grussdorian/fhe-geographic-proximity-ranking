# ==========================================
# setup_keys.py
# Generate shared FHE keys for all clients
# Run this ONCE before starting server/clients
# ==========================================

from openfhe import *
import os

KEYS_DIR = "fhe_keys"

def setup():
    os.makedirs(KEYS_DIR, exist_ok=True)

    # Generate crypto context
    # Pairwise scoring depth budget:
    #   - Distance: 1 ct-ct mult
    #   - Selector polynomial: 2 ct-ct + 1 pt-ct = 3 levels
    #   - Score × ID: 1 ct-ct mult
    #   - Total: 5 levels + 2 margin = 7
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(7)
    params.SetScalingModSize(50)
    params.SetFirstModSize(60)
    params.SetBatchSize(32)
    params.SetSecurityLevel(HEStd_128_classic)

    cc = GenCryptoContext(params)
    cc.Enable(PKE)
    cc.Enable(KEYSWITCH)
    cc.Enable(LEVELEDSHE)

    # Generate keys (no rotation keys needed — pairwise scoring
    # works slot-by-slot with one-hot vectors, no EvalRotate calls)
    keypair = cc.KeyGen()
    cc.EvalMultKeyGen(keypair.secretKey)

    # Serialize everything
    if not SerializeToFile(f"{KEYS_DIR}/cryptocontext.bin", cc, BINARY):
        raise RuntimeError("Failed to serialize crypto context")

    if not SerializeToFile(f"{KEYS_DIR}/publickey.bin", keypair.publicKey, BINARY):
        raise RuntimeError("Failed to serialize public key")

    if not SerializeToFile(f"{KEYS_DIR}/secretkey.bin", keypair.secretKey, BINARY):
        raise RuntimeError("Failed to serialize secret key")

    if not cc.SerializeEvalMultKey(f"{KEYS_DIR}/evalmultkey.bin", BINARY):
        raise RuntimeError("Failed to serialize eval mult key")

    print("Keys generated and saved to", KEYS_DIR)
    print("Files created:")
    for f in os.listdir(KEYS_DIR):
        print(f"  - {f}")


if __name__ == "__main__":
    setup()

# ==========================================
# setup_keys.py
# Generate the crypto context (parameters only)
# for threshold FHE.
#
# In the threshold model, secret keys are generated
# per-party during the interactive key generation
# protocol. This script only creates and saves the
# shared crypto context with MULTIPARTY enabled.
#
# Run this ONCE before starting server/clients.
# ==========================================

from openfhe import *
import os

KEYS_DIR = "fhe_keys"


def setup():
    os.makedirs(KEYS_DIR, exist_ok=True)

    # Generate crypto context
    # Pairwise scoring depth budget:
    #   - Distance: 1 ct-ct mult  (done by clients)
    #   - Selector polynomial: 2 ct-ct + 1 pt-ct = 3 levels  (server)
    #   - Score × ID: 1 ct-ct mult  (server)
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
    cc.Enable(ADVANCEDSHE)
    cc.Enable(MULTIPARTY)

    # Serialize just the context — keys are generated interactively
    if not SerializeToFile(f"{KEYS_DIR}/cryptocontext.bin", cc, BINARY):
        raise RuntimeError("Failed to serialize crypto context")

    print("Crypto context generated and saved to", KEYS_DIR)
    print(f"  MultiplicativeDepth = 7")
    print(f"  MULTIPARTY enabled (threshold FHE)")
    print(f"  BatchSize = 32")
    print()
    print("Next: Start the server, then have clients join for")
    print("      interactive threshold key generation.")


if __name__ == "__main__":
    setup()

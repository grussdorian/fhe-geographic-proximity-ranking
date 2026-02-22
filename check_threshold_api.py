#!/usr/bin/env python3.13
"""Check what threshold/multi-party FHE APIs are available in OpenFHE."""

import openfhe

# 1. CryptoContext methods
cc_methods = sorted([m for m in dir(openfhe.CryptoContext) if not m.startswith('_')])

# 2. Top-level openfhe attributes
top_attrs = sorted([a for a in dir(openfhe) if not a.startswith('_')])

# 3. Filter for threshold-related
threshold_kw = ['multi', 'thresh', 'party', 'share', 'joint', 'lead',
                'partial', 'fusion', 'decryptmain', 'decryptlead']

threshold_cc = [m for m in cc_methods if any(k in m.lower() for k in threshold_kw)]
threshold_top = [a for a in top_attrs if any(k in a.lower() for k in threshold_kw)]

with open('./openfhe_api_check.txt', 'w') as f:
    f.write("=== THRESHOLD-RELATED CC METHODS ===\n")
    for m in threshold_cc:
        f.write(f"  {m}\n")
    f.write(f"\n=== THRESHOLD-RELATED TOP-LEVEL ===\n")
    for a in threshold_top:
        f.write(f"  {a}\n")
    f.write(f"\n=== ALL CC METHODS ({len(cc_methods)}) ===\n")
    for m in cc_methods:
        f.write(f"  {m}\n")
    f.write(f"\n=== ALL TOP-LEVEL ({len(top_attrs)}) ===\n")
    for a in top_attrs:
        f.write(f"  {a}\n")

print(f"Wrote {len(cc_methods)} CC methods, {len(top_attrs)} top-level attrs to ./openfhe_api_check.txt")
print(f"Threshold CC methods: {threshold_cc}")
print(f"Threshold top-level: {threshold_top}")

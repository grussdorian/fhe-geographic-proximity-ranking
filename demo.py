import tenseal as ts

context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.generate_galois_keys()
context.global_scale = 2**40

secret_key = context.secret_key()
public_context = context.copy()
public_context.make_context_public()

scale_factor = 42.7

query_plain = [37.7749, -122.4194]
query_enc = ts.ckks_vector(context, query_plain)

members_plain = [
    [37.7755, -122.4190],
    [40.7128, -74.0060],
    [34.0522, -118.2437]
]
members_enc = [ts.ckks_vector(context, p) for p in members_plain]

def compute_enc_scaled_distances(query_enc, members_enc, scale_factor, context):
    enc_scaled_dists = []
    for m_enc in members_enc:
        diff_enc = query_enc - m_enc
        sq_diff_enc = diff_enc * diff_enc
        dist_sq_enc = sq_diff_enc.sum()           # vector sum of the two squared differences
        scaled_enc = dist_sq_enc * scale_factor
        enc_scaled_dists.append(scaled_enc)
    return enc_scaled_dists

query_enc.link_context(public_context)
for m in members_enc:
    m.link_context(public_context)

enc_scaled_dists = compute_enc_scaled_distances(query_enc, members_enc, scale_factor, public_context)

dec_scaled_dists = [round(d.decrypt(secret_key)[0], 6) for d in enc_scaled_dists]

ranked_indices = sorted(range(len(dec_scaled_dists)), key=lambda i: dec_scaled_dists[i])
top_index = ranked_indices[0]
top_scaled_value = dec_scaled_dists[top_index]

print("Decrypted scaled metrics:", dec_scaled_dists)
print("Ranked order (indices):", ranked_indices)
print("Closest member index:", top_index)
print("Its scaled metric:", top_scaled_value)
print("Client sees only: Closest member at index", top_index)
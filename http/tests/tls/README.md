# HTTPS test fixtures

ECDSA P-256 certificates for the hermetic `tests/https.rs` servers.
**Test-only material** — the private keys are deliberately public.

Each identity is a proper two-tier chain — a CA that signs a `CA:FALSE` leaf —
because rustls-webpki rejects a `CA:TRUE` certificate presented as the end
entity (`CaUsedAsEndEntity`), so a plain `openssl req -x509` self-signed cert
cannot be served.

- `cert_a.pem` — identity **A**'s *CA* as a PEM "CA bundle": the tests point
  `SSL_CERT_FILE` at it, so `httpsrc` trusts exactly A.
- `cert_a.der` / `key_a.der` — identity A's *leaf* for the rustls test server
  (certificate DER + PKCS#8 key DER).
- `cert_b.der` / `key_b.der` — identity **B**'s leaf, signed by a CA that is
  *not* in the bundle: the untrusted-certificate test serves B and must fail
  verification.

Leaf SANs cover `DNS:localhost` and `IP:127.0.0.1` (the tests dial
`https://127.0.0.1:<port>/`, verified against the IP SAN). Validity is 100
years, so the fixtures don't rot. Regenerate with:

```sh
for id in a b; do
  openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 -out ca_$id.key
  openssl req -x509 -key ca_$id.key -out ca_$id.pem -days 36500 \
    -subj "/CN=profluens test CA $id" \
    -addext "basicConstraints=critical,CA:TRUE" -addext "keyUsage=keyCertSign"
  openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 -out leaf_$id.key
  openssl req -new -key leaf_$id.key -out leaf_$id.csr \
    -subj "/CN=profluens test leaf $id"
  printf "subjectAltName=DNS:localhost,IP:127.0.0.1\nbasicConstraints=CA:FALSE\nkeyUsage=digitalSignature\nextendedKeyUsage=serverAuth\n" > ext_$id.cnf
  openssl x509 -req -in leaf_$id.csr -CA ca_$id.pem -CAkey ca_$id.key \
    -CAcreateserial -out leaf_$id.pem -days 36500 -extfile ext_$id.cnf
  openssl x509 -in leaf_$id.pem -outform der -out cert_$id.der
  openssl pkcs8 -topk8 -nocrypt -in leaf_$id.key -outform der -out key_$id.der
done
cp ca_a.pem cert_a.pem
rm ca_*.key ca_*.pem leaf_* ext_*.cnf *.srl   # keep only what the tests read
```

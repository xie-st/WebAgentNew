# Software Bill of Materials (SBOM)

effGen generates a [CycloneDX](https://cyclonedx.org/) SBOM on every release. This documents every Python package bundled in the release, enabling supply chain auditing and vulnerability tracking.

---

## What Is an SBOM?

A Software Bill of Materials (SBOM) is a formal record containing the details and supply chain relationships of the components used in building software. For effGen, this means every Python package installed in the runtime environment.

---

## Format

effGen uses **CycloneDX JSON v1.5**, the industry-standard SBOM format supported by:

- GitHub's dependency graph
- NIST NTIA SBOM compliance frameworks
- Most commercial SCA (Software Composition Analysis) tools
- OSV/NVD vulnerability databases

---

## Generating the SBOM

### Quick generation (current environment)

```bash
pip install cyclonedx-bom
cyclonedx-py environment \
  --of json \
  --sv 1.5 \
  -o sbom.cdx.json \
  --pyproject pyproject.toml
```

### CI Generation

The SBOM is automatically generated in CI (`.github/workflows/sbom.yml`):
- On every push to `main`.
- On every release tag (`v*`).
- Uploaded as a GitHub Actions artifact (retained 90 days).
- Attached to GitHub Releases as `sbom.cdx.json`.

---

## SBOM Structure

The CycloneDX JSON SBOM includes:

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.5",
  "serialNumber": "urn:uuid:...",
  "metadata": {
    "timestamp": "...",
    "tools": [{"vendor": "CycloneDX", "name": "cyclonedx-bom"}],
    "component": {"name": "effgen", "version": "0.2.x", "type": "library"}
  },
  "components": [
    {
      "type": "library",
      "name": "fastapi",
      "version": "0.136.1",
      "purl": "pkg:pypi/fastapi@0.136.1",
      "licenses": [{"license": {"id": "MIT"}}]
    }
    // ... 370+ more components
  ]
}
```

Key fields per component:
- `name` — Package name
- `version` — Installed version
- `purl` — Package URL for cross-referencing vulnerability databases
- `licenses` — License information (SPDX identifiers where available)
- `hashes` — SHA-256/MD5 hashes of the package files

---

## Using the SBOM

### Vulnerability Scanning

```bash
# Using grype (Anchore)
grype sbom:sbom.cdx.json

# Using osv-scanner (Google)
osv-scanner --sbom sbom.cdx.json

# Using trivy
trivy sbom sbom.cdx.json
```

### License Compliance

```bash
# Extract all licenses from the SBOM
python3 - << 'EOF'
import json
with open("sbom.cdx.json") as f:
    sbom = json.load(f)
licenses = set()
for comp in sbom.get("components", []):
    for lic in comp.get("licenses", []):
        if "id" in lic.get("license", {}):
            licenses.add(lic["license"]["id"])
for l in sorted(licenses):
    print(l)
EOF
```

### Integration with GitHub Dependency Graph

Upload the SBOM to GitHub's dependency graph via the API:

```bash
gh api \
  --method POST \
  -H "Accept: application/vnd.github+json" \
  /repos/OWNER/REPO/dependency-graph/snapshots \
  --input sbom.cdx.json
```

---

## Validation

effGen validates the SBOM on generation:

1. `bomFormat` must be `"CycloneDX"`.
2. `specVersion` must be `>= 1.4`.
3. `components` must be non-empty.
4. Core runtime deps (`fastapi`, `pydantic`, `openai`, `requests`, etc.) must appear.

Tests in `tests/security/test_sbom.py` enforce these invariants in CI.

---

## Release Artifacts

Every effGen release includes:
- `sbom.cdx.json` — Full CycloneDX JSON SBOM (attached to GitHub Release)
- Uploaded as CI artifact for every main-branch build

Future: wheel signing with Sigstore/cosign for supply chain integrity.

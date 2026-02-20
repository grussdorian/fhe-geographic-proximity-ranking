# Private proximity ranking using homomorphic encryption

Dependency - OpenFHE

```bash
python test_dense_urban.py
python test_fhe_proximity.py
```

## Run client and server on different terminals

### Terminal 1

```bash
python server.py
```

### Terminal 2

```bash
python client.py
```

## Testing and Logging

The test suites `test_dense_urban.py` and `test_fhe_proximity.py` support an optional `--log` flag to generate a comprehensive failure report when tests fail or encounter errors.

### Running Tests

```bash
# Normal run
python test_dense_urban.py

# Run with failure logging enabled
python test_dense_urban.py --log
```

### Logging Feature
- **Flag**: `--log`
- **Behavior**: If `--log` is present and any test **fails** or has an **error**, a detailed report is generated in the `logs/` directory.
- **Report Content**: Includes timestamp, device information (OS, Architecture, Python version), and the specific tracebacks/details for each failure.
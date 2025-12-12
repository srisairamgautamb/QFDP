#!/usr/bin/env python3
"""Check available IBM Quantum backends and their queue status"""

from qiskit_ibm_runtime import QiskitRuntimeService

print("Connecting to IBM Quantum...")
service = QiskitRuntimeService()

print("\nAvailable Backends (operational, not simulators):\n")
print(f"{'Backend':<20} {'Qubits':<8} {'Queue':<8} {'Status'}")
print("="*70)

backends = service.backends(simulator=False, operational=True)

if not backends:
    print("No operational backends available")
else:
    for backend in backends:
        try:
            status = backend.status()
            queue = status.pending_jobs if hasattr(status, 'pending_jobs') else 'N/A'
            operational = "✅" if status.operational else "❌"
            
            print(f"{backend.name:<20} {backend.num_qubits:<8} {queue:<8} {operational}")
        except Exception as e:
            print(f"{backend.name:<20} {backend.num_qubits:<8} {'ERROR':<8} ❌")

print("\nRecommendation: Use backend with lowest queue count")
print("\nTo use a different backend, modify test_hardware_extended.py:")
print("  backend = service.backend('BACKEND_NAME_HERE')")

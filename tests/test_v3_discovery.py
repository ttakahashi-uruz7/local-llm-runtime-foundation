from fastapi.testclient import TestClient

from runtime_foundation import HostProfile, RuntimeCore
from runtime_foundation.adapters.mock import MockAdapter
from runtime_foundation.service import create_app


def test_versioned_engine_capability_and_host_discovery_endpoints() -> None:
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": MockAdapter()})
    client = TestClient(create_app(core))

    engines = client.get("/v3/engines")
    capability = client.get("/v3/engines/mock/capability")
    host = client.get("/v3/host")

    assert engines.status_code == capability.status_code == host.status_code == 200
    assert engines.json()["engines"][0]["schema_version"] == "runtime-foundation.engine-capability.v3"
    assert capability.json()["contract_version"] == "runtime-foundation.contract.v3"
    assert host.json()["execution_capabilities"]["host_metal"] == "unsupported"

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SHA = '5a8f318629bbb6fcc4f0131164ab6088299cac9eeec44a76463a32f37baa3a59'


def test_adapter_distribution_metadata_is_consistent():
    manifest = json.loads(
        (ROOT / 'models/hr-orchestrator/manifest.json').read_text(encoding='utf-8')
    )
    distribution = manifest['adapter_distribution']

    assert manifest['adapter_sha256'] == EXPECTED_SHA
    assert distribution['provider'] == 'huggingface_hub'
    assert distribution['repository_env'] == 'ADAPTER_HF_REPO'
    assert distribution['revision_env'] == 'ADAPTER_HF_REVISION'
    assert set(distribution['required_files']) == {
        'adapter_config.json',
        'adapter_model.safetensors',
    }


def test_installers_download_and_verify_before_docker_start():
    linux = (ROOT / 'install.sh').read_text(encoding='utf-8')
    windows = (ROOT / 'install.ps1').read_text(encoding='utf-8')
    env_example = (ROOT / '.env.example').read_text(encoding='utf-8')

    assert linux.index('scripts/ensure_adapter.sh') < linux.index('docker compose pull')
    assert windows.index('scripts\\Ensure-Adapter.ps1') < windows.index('compose pull')
    assert 'ADAPTER_HF_REPO=' in env_example
    assert 'ADAPTER_HF_REVISION=main' in env_example


def test_checked_in_adapter_matches_approved_hash_when_present():
    adapter = ROOT / 'models/hr-orchestrator/adapter_model.safetensors'
    if not adapter.exists():
        return

    actual = hashlib.sha256(adapter.read_bytes()).hexdigest()
    assert actual == EXPECTED_SHA


def test_weight_is_ignored_by_git_configuration():
    ignore = (ROOT / '.gitignore').read_text(encoding='utf-8')
    assert 'models/hr-orchestrator/*' in ignore
    assert '!models/hr-orchestrator/adapter_model.safetensors' not in ignore

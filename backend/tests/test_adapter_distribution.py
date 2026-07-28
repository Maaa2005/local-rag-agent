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
    assert distribution['repository'] == 'yuu0617/hr-orchestrator'
    assert distribution['repository_env'] == 'ADAPTER_HF_REPO'
    assert distribution['revision_env'] == 'ADAPTER_HF_REVISION'
    assert set(distribution['required_files']) == {
        'adapter_config.json',
        'adapter_model.safetensors',
        'LICENSE',
        'NOTICE',
    }


def test_installers_download_and_verify_before_docker_start():
    linux = (ROOT / 'install.sh').read_text(encoding='utf-8')
    windows = (ROOT / 'install.ps1').read_text(encoding='utf-8')
    env_example = (ROOT / '.env.example').read_text(encoding='utf-8')

    assert linux.index('scripts/ensure_adapter.sh') < linux.index('docker compose pull')
    assert windows.index('scripts\\Ensure-Adapter.ps1') < windows.index('compose pull')
    assert 'ADAPTER_HF_REPO=' in env_example
    assert 'ADAPTER_HF_REPO=yuu0617/hr-orchestrator' in env_example
    assert 'ADAPTER_HF_REVISION=26e5631a7750c3c27d032d8fa375dc3f77917b1d' in env_example


def test_public_adapter_has_license_and_attribution_files():
    model_dir = ROOT / 'models/hr-orchestrator'
    card = (model_dir / 'README.md').read_text(encoding='utf-8')
    license_text = (model_dir / 'LICENSE').read_text(encoding='utf-8')
    notice = (model_dir / 'NOTICE').read_text(encoding='utf-8')

    assert 'license: apache-2.0' in card
    assert 'base_model: unsloth/gemma-4-E2B-it-unsloth-bnb-4bit' in card
    assert 'Apache License' in license_text
    assert 'Version 2.0, January 2004' in license_text
    assert 'google/gemma-4-E2B-it' in notice
    assert 'Modification notice:' in notice


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

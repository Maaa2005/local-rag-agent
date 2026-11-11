"""deploy/compose.yml の静的要件検査（N3・N5・N6 の配置面）。実際の分離は deploy/isolation_test.sh で試す。"""
from __future__ import annotations

import posixpath
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
COMPOSE = DEPLOY / "compose.yml"

INTERNAL_SERVICES = {"internal-app", "qdrant", "vllm", "judge-laya", "judge-clef"}
INTERNAL_VOLUMES = {"docs", "internal-data", "models", "qdrant-data"}


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def services(compose: dict) -> dict:
    return compose["services"]


def _networks(svc: dict) -> set[str]:
    n = svc.get("networks", {})
    return set(n) if isinstance(n, (list, dict)) else set()


def _volume_sources(svc: dict) -> list[str]:
    out = []
    for v in svc.get("volumes", []):
        out.append(v["source"] if isinstance(v, dict) else str(v).split(":", 1)[0])
    return out


def test_internal_network_is_internal(compose):
    assert compose["networks"]["internal"]["internal"] is True


def test_internal_services_only_on_internal_network(services):
    for name in INTERNAL_SERVICES:
        assert _networks(services[name]) == {"internal"}, name


def test_gateway_not_on_internal_network(services):
    nets = _networks(services["gateway"])
    assert nets == {"egress"}
    assert "network_mode" not in services["gateway"]


def test_egress_network_has_no_internal_service(services):
    for name, svc in services.items():
        if name != "gateway":
            assert "egress" not in _networks(svc), name


def test_frontend_publish_network_has_no_gateway(services):
    assert "frontend-publish" not in _networks(services["gateway"])
    for name, svc in services.items():
        if "frontend-publish" in _networks(svc):
            assert name == "frontend", name


def test_gateway_has_no_internal_volume_or_docker_socket(services):
    srcs = _volume_sources(services["gateway"])
    assert not INTERNAL_VOLUMES & set(srcs)
    assert not any("docker.sock" in s for s in srcs)
    assert set(srcs) == {"gateway-data", "gateway-sock"}


def test_no_service_mounts_docker_socket(services):
    for name, svc in services.items():
        assert not any("docker.sock" in s for s in _volume_sources(svc)), name


def test_gateway_sock_shared_only_by_gateway_and_internal_app(services):
    users = {n for n, s in services.items() if "gateway-sock" in _volume_sources(s)}
    assert users == {"gateway", "internal-app"}


def test_gateway_data_only_in_gateway(services):
    users = {n for n, s in services.items() if "gateway-data" in _volume_sources(s)}
    assert users == {"gateway"}


def test_only_frontend_publishes_ports_on_loopback(services):
    for name, svc in services.items():
        if name == "frontend":
            for p in svc["ports"]:
                assert str(p).startswith("127.0.0.1:"), p
        else:
            assert "ports" not in svc, name
            assert "expose" not in svc, name


def test_secrets_only_in_gateway(compose, services):
    for name, svc in services.items():
        if name == "gateway":
            assert set(svc["secrets"]) == {"anthropic_api_key", "openai_api_key"}
        else:
            assert "secrets" not in svc, name
    for sec in compose["secrets"].values():
        assert "file" in sec


def test_no_api_key_in_environment(services):
    for name, svc in services.items():
        env = svc.get("environment", {}) or {}
        keys = env.keys() if isinstance(env, dict) else [e.split("=", 1)[0] for e in env]
        for k in keys:
            assert not (k.endswith("_API_KEY") or k.endswith("_TOKEN")), (name, k)


@pytest.mark.parametrize("name", sorted(INTERNAL_SERVICES | {"gateway", "frontend"}))
def test_hardening(services, name):
    svc = services[name]
    assert svc.get("read_only") is True
    assert svc.get("cap_drop") == ["ALL"]
    assert "no-new-privileges:true" in svc.get("security_opt", [])
    user = str(svc.get("user", ""))
    assert user and not user.startswith("0") and not user.startswith("root")
    assert not svc.get("privileged", False)


def test_clef_profile_stops_vllm(services):
    assert "clef" not in services["vllm"].get("profiles", [])
    assert services["judge-clef"]["profiles"] == ["clef"]
    assert services["judge-laya"]["profiles"] == ["laya"]
    assert "laya" in services["vllm"]["profiles"]


@pytest.mark.parametrize("name", ["vllm", "judge-laya", "judge-clef", "internal-app"])
def test_offline_model_settings(services, name):
    env = services[name]["environment"]
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert "HUGGING_FACE_HUB_TOKEN" not in env and "HF_TOKEN" not in env


def test_models_mounted_read_only(services):
    for name, svc in services.items():
        for v in svc.get("volumes", []):
            if str(v).startswith("models:"):
                assert str(v).endswith(":ro"), (name, v)


def test_vllm_uses_local_model_path(services):
    assert "--model /models/" in services["vllm"]["command"]


def test_vllm_does_not_force_quantization(services):
    # 既定モデルは compressed-tensors。--quantization awq を付けると vLLM が起動時に止まる（固定版 vLLM 0.25.1 のソースで確認）
    assert "--quantization" not in services["vllm"]["command"]


def test_gateway_dockerfile_copies_only_common_and_gateway():
    text = (DEPLOY / "gateway.Dockerfile").read_text(encoding="utf-8")
    copies = [ln.split()[1] for ln in text.splitlines() if ln.startswith("COPY")]
    assert sorted(copies) == ["common/", "gateway/"]
    assert "FROM python:3.11-slim" in text
    users = [ln for ln in text.splitlines() if ln.startswith("USER")]
    assert users and "root" not in users[-1] and not users[-1].split()[1].startswith("0")


def _dockerfile_lines(svc: dict) -> list[str]:
    text = (DEPLOY / Path(svc["build"]["dockerfile"]).name).read_text(encoding="utf-8")
    return [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]


@pytest.mark.parametrize("name", ["gateway", "internal-app"])
def test_writable_volumes_owned_by_service_user_in_image(services, name):
    # 名前付き volume は空のときイメージ内の同じパスの所有者を写す。パスが無いと root 所有 755 になり、
    # 非 root のサービスが書けない（10/26 internal-app が監査 DB を作れなかった）
    svc = services[name]
    body = "\n".join(_dockerfile_lines(svc))
    for v in svc["volumes"]:
        src, dst, *mode = str(v).split(":")
        if mode == ["ro"] or (name != "gateway" and src == "gateway-sock"):
            continue
        assert re.search(rf"chown {svc['user']} [^&\n]*{re.escape(dst)}(\s|$)", body), (name, dst)


def test_internal_app_dockerfile_copies_only_common_and_orchestrator(services):
    lines = _dockerfile_lines(services["internal-app"])
    copies = [ln.split()[1] for ln in lines if ln.startswith("COPY")]
    assert sorted(copies) == ["common/", "orchestrator/"]
    # gateway-sock の所有者は gateway イメージで決める。internal-app 側で作ると先に起動した側が写る
    assert not any("/run/gateway" in ln for ln in lines)
    users = [ln for ln in lines if ln.startswith("USER")]
    assert users and users[-1].split()[1] == services["internal-app"]["user"]


# ---- vllm の配置（Issue #14・2026-11-04） ----

VLLM_DIGEST = "sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089"


def test_vllm_image_is_pinned_by_digest(services):
    image = services["vllm"]["image"]
    assert image == f"vllm/vllm-openai:v0.25.1@{VLLM_DIGEST}"
    assert re.fullmatch(r"[^@]+@sha256:[0-9a-f]{64}", image)


def _is_tmp_path(value: object) -> bool:
    # 前方一致では /tmp-cache や /tmp/../root/cache を通してしまう。正規化してから /tmp 自身か配下かを見る
    if not isinstance(value, str):
        return False
    path = posixpath.normpath(value)
    return path == "/tmp" or path.startswith("/tmp/")


@pytest.mark.parametrize("value", ["/tmp", "/tmp/", "/tmp/.cache", "/tmp/a/../b"])
def test_is_tmp_path_accepts_tmp_and_children(value):
    assert _is_tmp_path(value)


@pytest.mark.parametrize("value", ["/tmp-cache", "/tmp/../root/cache", "/tmp/..", "tmp/x", "/root/tmp", "", None])
def test_is_tmp_path_rejects_outside_tmp(value):
    assert not _is_tmp_path(value)


def test_vllm_usage_stats_are_off(services):
    env = services["vllm"]["environment"]
    # 値は文字列の "1" の時だけ効く（"true" では効かない）
    for k in ("VLLM_NO_USAGE_STATS", "VLLM_DO_NOT_TRACK", "DO_NOT_TRACK", "HF_HUB_DISABLE_TELEMETRY"):
        assert env[k] == "1", k
    for k in ("HOME", "XDG_CACHE_HOME", "VLLM_CONFIG_ROOT", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR"):
        assert _is_tmp_path(env[k]), k


def test_vllm_runs_as_the_supported_non_root_user(services):
    svc = services["vllm"]
    # 公式の vllm-openai が非 root で支えるのは同梱の vllm（UID 2000・GID 0）だけ（v0.25.1 docs/deployment/docker.md）
    assert str(svc["user"]) == "2000:0"
    assert svc["cap_drop"] == ["ALL"] and svc["read_only"] is True
    others = {str(s.get("user", "")).split(":")[0] for n, s in services.items() if n != "vllm"}
    assert "2000" not in others


VLLM_TMPFS = "/tmp:exec,size=1g"


def _tmpfs_entries(svc: dict) -> list[str]:
    # compose は tmpfs の単一文字列も受ける。文字列のまま回すと 1 文字ずつの検査になる
    t = svc.get("tmpfs") or []
    return [t] if isinstance(t, str) else [str(x) for x in t]


def _tmpfs_exec_violations(services: dict) -> list[tuple[str, str]]:
    bad = []
    for name, svc in services.items():
        for t in _tmpfs_entries(svc):
            opts = t.split(":", 1)[1].split(",") if ":" in t else []
            if "exec" in opts and (name, t) != ("vllm", VLLM_TMPFS):
                bad.append((name, t))
    # 無いと Triton の .so を読めない。存在の検査がないとマウントの削除や noexec 化を見逃す
    vllm_tmp = [t for t in _tmpfs_entries(services["vllm"]) if posixpath.normpath(t.split(":", 1)[0]) == "/tmp"]
    if vllm_tmp != [VLLM_TMPFS]:
        bad.append(("vllm", f"/tmp は {VLLM_TMPFS} の 1 件だけのはず: {vllm_tmp}"))
    return bad


def test_only_vllm_tmp_is_exec(services):
    assert _tmpfs_exec_violations(services) == []


@pytest.mark.parametrize(
    "tmpfs",
    [None, [], "/tmp", ["/tmp"], ["/tmp:noexec,size=1g"], ["/tmp:size=1g"], [VLLM_TMPFS, "/tmp"], ["/var/tmp:exec,size=1g"]],
)
def test_tmpfs_check_rejects_broken_vllm_mount(tmpfs):
    vllm = {"tmpfs": tmpfs} if tmpfs is not None else {}
    assert _tmpfs_exec_violations({"vllm": vllm})


def test_tmpfs_check_rejects_exec_outside_vllm():
    assert _tmpfs_exec_violations({"vllm": {"tmpfs": [VLLM_TMPFS]}, "qdrant": {"tmpfs": "/tmp:exec"}})


def test_tmpfs_check_accepts_expected_mount():
    assert _tmpfs_exec_violations({"vllm": {"tmpfs": [VLLM_TMPFS]}, "qdrant": {"tmpfs": ["/tmp"]}}) == []


def test_vllm_text_only_and_single_sequence(services):
    cmd = services["vllm"]["command"].split()
    assert "--language-model-only" in cmd
    assert cmd[cmd.index("--max-num-seqs") + 1] == "1"
    assert "--quantization" not in cmd

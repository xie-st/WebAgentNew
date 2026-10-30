"""Helm chart tests for effGen.

Tests verify:
1. Chart structure and required files exist.
2. helm lint passes with 0 errors.
3. helm template produces valid YAML with required Kubernetes resources.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_DIR = REPO_ROOT / "deploy" / "k8s" / "helm" / "effgen"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _helm_available() -> bool:
    """Return True if helm binary is on PATH."""
    return shutil.which("helm") is not None or (Path.home() / ".local" / "bin" / "helm").exists()


def _helm_bin() -> str:
    """Return path to helm binary."""
    local_helm = Path.home() / ".local" / "bin" / "helm"
    if local_helm.exists():
        return str(local_helm)
    h = shutil.which("helm")
    if h:
        return h
    pytest.skip("helm binary not found")


HELM_AVAILABLE = _helm_available()
requires_helm = pytest.mark.skipif(not HELM_AVAILABLE, reason="helm not installed")


def _kubeconform_bin() -> str | None:
    """Return path to a kubeconform binary if available, else None."""
    local = Path.home() / ".local" / "bin" / "kubeconform"
    if local.exists():
        return str(local)
    return shutil.which("kubeconform")


KUBECONFORM_BIN = _kubeconform_bin()
requires_kubeconform = pytest.mark.skipif(
    KUBECONFORM_BIN is None,
    reason="kubeconform not installed (offline K8s schema validation)",
)


# ---------------------------------------------------------------------------
# Chart structure tests — always run
# ---------------------------------------------------------------------------

class TestChartStructure:
    def test_chart_dir_exists(self) -> None:
        assert CHART_DIR.exists(), f"Helm chart directory not found: {CHART_DIR}"

    def test_chart_yaml_exists(self) -> None:
        assert (CHART_DIR / "Chart.yaml").exists()

    def test_values_yaml_exists(self) -> None:
        assert (CHART_DIR / "values.yaml").exists()

    def test_templates_dir_exists(self) -> None:
        assert (CHART_DIR / "templates").is_dir()

    def test_required_templates_present(self) -> None:
        templates = CHART_DIR / "templates"
        required = [
            "deployment.yaml",
            "service.yaml",
            "serviceaccount.yaml",
            "networkpolicy.yaml",
            "pdb.yaml",
            "hpa.yaml",
            "ingress.yaml",
            "_helpers.tpl",
        ]
        for fname in required:
            assert (templates / fname).exists(), f"Missing required template: {fname}"


class TestChartYaml:
    def setup_method(self) -> None:
        self.chart = yaml.safe_load((CHART_DIR / "Chart.yaml").read_text())

    def test_api_version_v2(self) -> None:
        assert self.chart["apiVersion"] == "v2"

    def test_name_effgen(self) -> None:
        assert self.chart["name"] == "effgen"

    def test_version_present(self) -> None:
        assert "version" in self.chart

    def test_app_version_present(self) -> None:
        assert "appVersion" in self.chart

    def test_description_present(self) -> None:
        assert "description" in self.chart and len(self.chart["description"]) > 5


class TestValuesYaml:
    def setup_method(self) -> None:
        self.values = yaml.safe_load((CHART_DIR / "values.yaml").read_text())

    def test_replica_count_at_least_2(self) -> None:
        assert self.values["replicaCount"] >= 2

    def test_autoscaling_enabled(self) -> None:
        assert self.values["autoscaling"]["enabled"] is True

    def test_hpa_min_replicas_2(self) -> None:
        assert self.values["autoscaling"]["minReplicas"] >= 2

    def test_hpa_custom_metric_present(self) -> None:
        custom = self.values["autoscaling"].get("customMetrics", [])
        metric_names = []
        for m in custom:
            if m.get("type") == "Pods":
                metric_names.append(m["pods"]["metric"]["name"])
        assert "effgen_model_call_latency_seconds" in metric_names, \
            "HPA must include effgen_model_call_latency_seconds custom metric"

    def test_network_policy_enabled(self) -> None:
        assert self.values["networkPolicy"]["enabled"] is True

    def test_pdb_enabled(self) -> None:
        assert self.values["podDisruptionBudget"]["enabled"] is True

    def test_pdb_min_available(self) -> None:
        assert self.values["podDisruptionBudget"]["minAvailable"] >= 1

    def test_non_root_security_context(self) -> None:
        ctx = self.values["podSecurityContext"]
        assert ctx.get("runAsNonRoot") is True
        uid = ctx.get("runAsUser", 0)
        assert uid != 0, f"runAsUser should not be root (0), got {uid}"

    def test_dev_mode_off_by_default(self) -> None:
        env = self.values.get("env", {})
        assert str(env.get("EFFGEN_DEV_MODE", "0")) == "0", \
            "EFFGEN_DEV_MODE must be '0' (auth on) by default"

    def test_resource_requests_present(self) -> None:
        res = self.values["resources"]
        assert "requests" in res
        assert "cpu" in res["requests"]
        assert "memory" in res["requests"]

    def test_health_probes_present(self) -> None:
        # Liveness uses /livez, readiness uses /readyz — the conventional K8s
        # probe paths, all served (and auth-exempt) by the API.
        assert self.values["livenessProbe"]["httpGet"]["path"] == "/livez"
        assert self.values["readinessProbe"]["httpGet"]["path"] == "/readyz"


# ---------------------------------------------------------------------------
# Helm CLI tests — require helm installed
# ---------------------------------------------------------------------------

@requires_helm
class TestHelmLint:
    def test_helm_lint_zero_errors(self) -> None:
        result = subprocess.run(
            [_helm_bin(), "lint", str(CHART_DIR)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, \
            f"helm lint failed:\n{result.stdout}\n{result.stderr}"
        assert "0 chart(s) failed" in result.stdout, \
            f"helm lint reported failures:\n{result.stdout}"

    def test_helm_lint_no_errors_keyword(self) -> None:
        result = subprocess.run(
            [_helm_bin(), "lint", str(CHART_DIR)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # "[ERROR]" would be fatal; "[WARNING]" is acceptable
        assert "[ERROR]" not in result.stdout, \
            f"helm lint ERROR found:\n{result.stdout}"


@requires_helm
class TestHelmTemplate:
    """Verify helm template output contains required Kubernetes resources."""

    @pytest.fixture(scope="class")
    def rendered(self) -> list[dict]:  # type: ignore[override]
        """Run helm template once and parse all YAML documents."""
        result = subprocess.run(
            [_helm_bin(), "template", "effgen-ci", str(CHART_DIR)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, \
            f"helm template failed:\n{result.stdout}\n{result.stderr}"
        docs = list(yaml.safe_load_all(result.stdout))
        return [d for d in docs if d]  # filter None docs

    def test_deployment_present(self, rendered: list[dict]) -> None:
        kinds = [d.get("kind") for d in rendered]
        assert "Deployment" in kinds

    def test_service_present(self, rendered: list[dict]) -> None:
        kinds = [d.get("kind") for d in rendered]
        assert "Service" in kinds

    def test_serviceaccount_present(self, rendered: list[dict]) -> None:
        kinds = [d.get("kind") for d in rendered]
        assert "ServiceAccount" in kinds

    def test_hpa_present(self, rendered: list[dict]) -> None:
        kinds = [d.get("kind") for d in rendered]
        assert "HorizontalPodAutoscaler" in kinds

    def test_pdb_present(self, rendered: list[dict]) -> None:
        kinds = [d.get("kind") for d in rendered]
        assert "PodDisruptionBudget" in kinds

    def test_networkpolicy_present(self, rendered: list[dict]) -> None:
        kinds = [d.get("kind") for d in rendered]
        assert "NetworkPolicy" in kinds

    def test_deployment_replicas_not_set_when_hpa(self, rendered: list[dict]) -> None:
        """When HPA is enabled, Deployment should not set replicas (HPA controls it)."""
        deployments = [d for d in rendered if d.get("kind") == "Deployment"]
        assert deployments, "No Deployment found"
        dep = deployments[0]
        spec = dep.get("spec", {})
        # replicas key should be absent when HPA is active
        assert "replicas" not in spec, \
            "Deployment must not set .spec.replicas when HPA is enabled"

    def test_deployment_has_health_probes(self, rendered: list[dict]) -> None:
        dep = next(d for d in rendered if d.get("kind") == "Deployment")
        container = dep["spec"]["template"]["spec"]["containers"][0]
        assert "livenessProbe" in container
        assert "readinessProbe" in container
        assert container["livenessProbe"]["httpGet"]["path"] == "/livez"
        assert container["readinessProbe"]["httpGet"]["path"] == "/readyz"

    def test_deployment_non_root_security(self, rendered: list[dict]) -> None:
        dep = next(d for d in rendered if d.get("kind") == "Deployment")
        pod_spec = dep["spec"]["template"]["spec"]
        sec = pod_spec.get("securityContext", {})
        assert sec.get("runAsNonRoot") is True

    def test_hpa_custom_metric(self, rendered: list[dict]) -> None:
        hpa = next(d for d in rendered if d.get("kind") == "HorizontalPodAutoscaler")
        metrics = hpa["spec"]["metrics"]
        custom_names = [
            m["pods"]["metric"]["name"]
            for m in metrics
            if m.get("type") == "Pods"
        ]
        assert "effgen_model_call_latency_seconds" in custom_names, \
            f"HPA missing custom metric. Found: {custom_names}"

    def test_networkpolicy_restricts_ingress(self, rendered: list[dict]) -> None:
        np = next(d for d in rendered if d.get("kind") == "NetworkPolicy")
        assert "Ingress" in np["spec"]["policyTypes"]

    def test_networkpolicy_allows_egress_dns(self, rendered: list[dict]) -> None:
        np = next(d for d in rendered if d.get("kind") == "NetworkPolicy")
        assert "Egress" in np["spec"]["policyTypes"]
        egress = np["spec"].get("egress", [])
        dns_ports = [
            rule for rule in egress
            for port in rule.get("ports", [])
            if port.get("port") == 53
        ]
        assert dns_ports, "NetworkPolicy egress must allow DNS (port 53)"

    def test_prometheus_scrape_annotation_is_reachable(self, rendered: list[dict]) -> None:
        """A pod that advertises a Prometheus scrape must actually serve it.

        The pod annotations point Prometheus at /metrics; if that endpoint
        stays auth-gated (the server default) every scrape returns 401.
        Either the annotation is dropped or EFFGEN_PUBLIC_METRICS must open
        the endpoint (the NetworkPolicy is the access control in that case).
        """
        dep = next(d for d in rendered if d.get("kind") == "Deployment")
        annotations = dep["spec"]["template"]["metadata"].get("annotations", {})
        if annotations.get("prometheus.io/scrape") != "true":
            return
        env_vars = dep["spec"]["template"]["spec"]["containers"][0]["env"]
        public_metrics = next(
            (e["value"] for e in env_vars if e["name"] == "EFFGEN_PUBLIC_METRICS"), None
        )
        assert public_metrics == "1", (
            "prometheus.io/scrape=true is set but EFFGEN_PUBLIC_METRICS is not "
            "'1' — Prometheus would get 401 from the auth-gated /metrics endpoint"
        )


@requires_helm
class TestHelmTemplatePersistence:
    """Persistence path must render a matching PVC for the audit-log volume."""

    def _render(self, *extra: str) -> list[dict]:
        result = subprocess.run(
            [_helm_bin(), "template", "effgen-ci", str(CHART_DIR), *extra],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, \
            f"helm template failed:\n{result.stdout}\n{result.stderr}"
        return [d for d in yaml.safe_load_all(result.stdout) if d]

    def test_pvc_rendered_when_persistence_enabled(self) -> None:
        docs = self._render("--set", "persistence.auditLog.enabled=true")
        pvcs = [d for d in docs if d.get("kind") == "PersistentVolumeClaim"]
        assert pvcs, "persistence.auditLog.enabled=true must render a PersistentVolumeClaim"
        # The Deployment claimName must match the PVC name (else pods hang Pending).
        dep = next(d for d in docs if d.get("kind") == "Deployment")
        volumes = dep["spec"]["template"]["spec"]["volumes"]
        claim_names = [
            v["persistentVolumeClaim"]["claimName"]
            for v in volumes
            if "persistentVolumeClaim" in v
        ]
        pvc_names = [p["metadata"]["name"] for p in pvcs]
        assert set(claim_names) <= set(pvc_names), \
            f"Deployment references PVC(s) {claim_names} but only {pvc_names} are created"

    def test_no_pvc_when_persistence_disabled(self) -> None:
        docs = self._render()  # defaults: persistence disabled
        pvcs = [d for d in docs if d.get("kind") == "PersistentVolumeClaim"]
        assert not pvcs, "No PVC should render when persistence.auditLog.enabled=false"
        dep = next(d for d in docs if d.get("kind") == "Deployment")
        volumes = dep["spec"]["template"]["spec"]["volumes"]
        audit = next(v for v in volumes if v["name"] == "audit-log")
        assert "emptyDir" in audit, "audit-log must fall back to emptyDir when persistence off"


@requires_helm
@requires_kubeconform
class TestHelmKubeconform:
    """Validate rendered manifests against offline Kubernetes JSON schemas.

    This covers the same intent as ``kubectl apply --dry-run=client`` (valid
    K8s YAML) without needing a live cluster: kubeconform validates each
    resource against the upstream OpenAPI-derived JSON schemas.
    """

    def _render(self, *extra: str) -> str:
        result = subprocess.run(
            [_helm_bin(), "template", "effgen-ci", str(CHART_DIR), *extra],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, \
            f"helm template failed:\n{result.stdout}\n{result.stderr}"
        return result.stdout

    def _validate(self, manifests: str) -> None:
        result = subprocess.run(
            [
                str(KUBECONFORM_BIN),
                "-strict",
                "-summary",
                "-kubernetes-version", "1.29.0",
            ],
            input=manifests,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, \
            f"kubeconform validation failed:\n{result.stdout}\n{result.stderr}"
        assert "Invalid: 0" in result.stdout and "Errors: 0" in result.stdout, \
            f"kubeconform reported invalid resources:\n{result.stdout}"

    def test_default_values_valid(self) -> None:
        self._validate(self._render())

    def test_full_feature_values_valid(self) -> None:
        self._validate(self._render(
            "--set", "ingress.enabled=true",
            "--set", "configMap.enabled=true",
            "--set", "configMap.data.FOO=bar",
            "--set", "secrets.CEREBRAS_API_KEY=dummy",
            "--set", "persistence.auditLog.enabled=true",
        ))


@requires_helm
class TestHelmTemplateDevMode:
    """Verify dev-mode override renders the warning annotation."""

    def test_dev_mode_values_override(self) -> None:
        result = subprocess.run(
            [
                _helm_bin(), "template", "effgen-ci", str(CHART_DIR),
                "--set", "env.EFFGEN_DEV_MODE=1",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0
        docs = [d for d in yaml.safe_load_all(result.stdout) if d]
        dep = next(d for d in docs if d.get("kind") == "Deployment")
        env_vars = dep["spec"]["template"]["spec"]["containers"][0]["env"]
        dev_mode = next(
            (e["value"] for e in env_vars if e["name"] == "EFFGEN_DEV_MODE"), None
        )
        assert dev_mode == "1"

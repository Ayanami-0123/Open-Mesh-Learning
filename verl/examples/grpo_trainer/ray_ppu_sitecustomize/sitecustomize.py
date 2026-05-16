"""Runtime patches for Ray dashboard on the local PPU image.

Ray's dashboard agent can mis-detect the PPU runtime as NVIDIA and call NVML
process-utilization APIs that are unsupported by HGGC. The dashboard agent is
fate-shared with raylet, so that stderr-level NVML failure kills the whole local
Ray node before verl workers start. Disable only dashboard GPU metrics here.
"""

try:
    from ray.dashboard.modules.reporter import gpu_providers

    def _disable_gpu_metrics(self):
        self._provider = None
        self._enable_metric_report = False
        self._initialized = True
        return False

    gpu_providers.GpuMetricProvider.initialize = _disable_gpu_metrics
    gpu_providers.GpuMetricProvider.get_gpu_usage = lambda self: []
    gpu_providers.GpuMetricProvider.get_provider_name = lambda self: None
    gpu_providers.GpuMetricProvider.is_metric_report_enabled = lambda self: False
except Exception:
    pass

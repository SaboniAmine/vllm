# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Energy measurement using CodeCarbon for vLLM inference.

Provides cached energy readings from CodeCarbon's background sampling.
The scheduler handles attribution by reading energy at request start/end.

Key insight: NVML has ~100ms update cycle on modern GPUs. Background
sampling at this rate gives zero critical-path overhead with no loss
in precision.

Environment Variables:
    VLLM_TRACK_ENERGY: Set to "1" to enable energy tracking
    VLLM_ENERGY_OUTPUT: Path to JSON file for step-level energy data
    CODECARBON_COUNTRY: ISO country code for carbon intensity (default: USA)
"""

import json
import os
import time
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from codecarbon import OfflineEmissionsTracker

logger = init_logger(__name__)

# Measurement interval for background sampling
_MEASURE_INTERVAL_SECS = 0.05  # 50ms


class EnergyMetrics:
    """Cached energy measurement using CodeCarbon's background sampling.

    Provides a single method to read the current total energy from cache.
    The scheduler handles per-request attribution.

    Usage:
        metrics = EnergyMetrics()

        # Read current energy (from cache, instant):
        energy_kwh = metrics.get_energy_kwh()
    """

    def __init__(self) -> None:
        """Initialize energy tracking based on environment variable."""
        self._enabled = os.environ.get("VLLM_TRACK_ENERGY", "0") == "1"
        self._tracker: "OfflineEmissionsTracker | None" = None

        # Accumulated totals for prefill vs decode energy
        self._total_prefill_energy_kwh: float = 0.0
        self._total_decode_energy_kwh: float = 0.0
        self._total_prefill_tokens: int = 0
        self._total_decode_tokens: int = 0

        # JSON output for analysis
        self._output_path = os.environ.get("VLLM_ENERGY_OUTPUT", "")
        self._step_history: list[dict[str, Any]] = []
        self._step_count: int = 0
        self._start_time: float = time.time()

        if self._enabled:
            self._init_tracker()

    def _init_tracker(self) -> None:
        """Initialize the CodeCarbon tracker with fast background sampling."""
        try:
            from codecarbon import OfflineEmissionsTracker

            country_code = os.environ.get("CODECARBON_COUNTRY", "USA")
            self._tracker = OfflineEmissionsTracker(
                country_iso_code=country_code,
                measure_power_secs=_MEASURE_INTERVAL_SECS,
                save_to_file=False,
                save_to_api=False,
                log_level="error",
            )
            self._tracker.start()
            logger.info(
                "Energy tracking enabled (interval=%dms, country=%s)",
                int(_MEASURE_INTERVAL_SECS * 1000),
                country_code,
            )
        except ImportError:
            logger.warning(
                "CodeCarbon not installed, energy tracking disabled. "
                "Install with: pip install codecarbon"
            )
            self._enabled = False
        except Exception as e:
            logger.warning("Failed to initialize energy tracker: %s", e)
            self._enabled = False

    def is_enabled(self) -> bool:
        """Return whether energy tracking is enabled and active."""
        return self._enabled and self._tracker is not None

    def get_energy_kwh(self) -> float:
        """Get current total energy from cache (no NVML call).

        Returns:
            Total energy consumed in kWh since tracker started
        """
        if not self.is_enabled():
            return 0.0
        return getattr(self._tracker._total_energy, "kWh", 0.0)

    def accumulate(
        self,
        prefill_energy: float,
        decode_energy: float,
        prefill_tokens: int,
        decode_tokens: int,
    ) -> None:
        """Accumulate prefill/decode energy totals.

        Args:
            prefill_energy: Energy attributed to prefill in this step (kWh)
            decode_energy: Energy attributed to decode in this step (kWh)
            prefill_tokens: Number of prefill tokens in this step
            decode_tokens: Number of decode tokens in this step
        """
        self._total_prefill_energy_kwh += prefill_energy
        self._total_decode_energy_kwh += decode_energy
        self._total_prefill_tokens += prefill_tokens
        self._total_decode_tokens += decode_tokens

        # Record step data for JSON output
        if self._output_path:
            self._step_count += 1
            step_data = {
                "step": self._step_count,
                "timestamp": time.time() - self._start_time,
                "step_energy_kwh": prefill_energy + decode_energy,
                "prefill_energy_kwh": prefill_energy,
                "decode_energy_kwh": decode_energy,
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "total_tokens": prefill_tokens + decode_tokens,
                # Running totals
                "cumulative_prefill_energy_kwh": self._total_prefill_energy_kwh,
                "cumulative_decode_energy_kwh": self._total_decode_energy_kwh,
                "cumulative_prefill_tokens": self._total_prefill_tokens,
                "cumulative_decode_tokens": self._total_decode_tokens,
            }
            self._step_history.append(step_data)

    def get_totals(self) -> tuple[float, float, int, int]:
        """Return accumulated prefill/decode energy totals.

        Returns:
            Tuple of (total_prefill_energy_kwh, total_decode_energy_kwh,
                      total_prefill_tokens, total_decode_tokens)
        """
        return (
            self._total_prefill_energy_kwh,
            self._total_decode_energy_kwh,
            self._total_prefill_tokens,
            self._total_decode_tokens,
        )

    def save_to_json(self) -> None:
        """Save step history to JSON file if output path is configured."""
        if not self._output_path or not self._step_history:
            return

        # Compute per-token energy metrics
        total_energy = self._total_prefill_energy_kwh + self._total_decode_energy_kwh
        total_tokens = self._total_prefill_tokens + self._total_decode_tokens

        output_data = {
            "summary": {
                "total_steps": self._step_count,
                "total_duration_seconds": time.time() - self._start_time,
                "total_energy_kwh": total_energy,
                "total_energy_wh": total_energy * 1000,
                "prefill": {
                    "energy_kwh": self._total_prefill_energy_kwh,
                    "energy_wh": self._total_prefill_energy_kwh * 1000,
                    "tokens": self._total_prefill_tokens,
                    "joules_per_token": (
                        (self._total_prefill_energy_kwh * 3600000)
                        / self._total_prefill_tokens
                        if self._total_prefill_tokens > 0 else 0.0
                    ),
                },
                "decode": {
                    "energy_kwh": self._total_decode_energy_kwh,
                    "energy_wh": self._total_decode_energy_kwh * 1000,
                    "tokens": self._total_decode_tokens,
                    "joules_per_token": (
                        (self._total_decode_energy_kwh * 3600000)
                        / self._total_decode_tokens
                        if self._total_decode_tokens > 0 else 0.0
                    ),
                },
                "total_tokens": total_tokens,
                "avg_joules_per_token": (
                    (total_energy * 3600000) / total_tokens
                    if total_tokens > 0 else 0.0
                ),
            },
            "steps": self._step_history,
        }

        try:
            with open(self._output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            logger.info("Energy stats saved to: %s", self._output_path)
        except Exception as e:
            logger.warning("Failed to save energy stats to %s: %s",
                          self._output_path, e)

    def shutdown(self) -> None:
        """Stop the energy tracker and clean up resources."""
        # Save JSON output before stopping
        self.save_to_json()

        if self._tracker:
            self._tracker.stop()
            self._tracker = None
        self._enabled = False

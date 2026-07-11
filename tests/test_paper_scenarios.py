import unittest

from adaptive_stream_agents.cli import _parse_agent_models
from adaptive_stream_agents.scenarios.conflict_resolution import (
    build_conflict_resolution_scenario,
)
from adaptive_stream_agents.scenarios.traffic_drift import (
    build_traffic_drift_scenario,
)


class PaperScenarioSmokeTest(unittest.TestCase):
    def test_conflict_scenario_builds_with_scalable_setting(self) -> None:
        scenario = build_conflict_resolution_scenario(num_agents=5, num_requests=3)

        self.assertEqual(len(scenario.agents), 5)
        self.assertEqual(len(scenario.requests), 3)

    def test_traffic_drift_scenario_builds(self) -> None:
        scenario = build_traffic_drift_scenario()

        self.assertEqual(scenario.initiator_id, "A0")
        self.assertEqual(len(scenario.windows), 30)

    def test_parse_agent_models_supports_mixed_llms(self) -> None:
        models = _parse_agent_models("A0=llama3,A1=phi,A4=gpt-oss:20b:cloud")

        self.assertEqual(models["A0"], "llama3")
        self.assertEqual(models["A1"], "phi")
        self.assertEqual(models["A4"], "gpt-oss:20b:cloud")


if __name__ == "__main__":
    unittest.main()

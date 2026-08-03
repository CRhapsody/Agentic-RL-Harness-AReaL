from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from jphrl.trajectory.areal_interaction_sidecar import (
    ArealInteractionAdapterError,
    InteractionBinding,
    build_interaction_adapter_sidecar,
    export_bound_training_sample_archive,
    interaction_id_for_model_call,
    model_call_id_for_interaction,
    validate_bound_training_sample_archive,
    validate_interaction_adapter_sidecar,
    write_bound_training_sample_archive,
    write_interaction_adapter_sidecar,
)


class FakeInteraction:
    def __init__(
        self,
        *,
        interaction_id: str,
        parent: FakeInteraction | None,
        chat_template_type: str,
        input_tokens: list[int],
        output_tokens: list[int],
        output_logprobs: list[float],
        output_versions: list[int],
        tensor_dict: dict[str, object],
    ) -> None:
        self.interaction_id = interaction_id
        self.parent = parent
        self.chat_template_type = chat_template_type
        self.model_response = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            output_versions=output_versions,
        )
        self._tensor_dict = tensor_dict

    def to_tensor_dict(self) -> dict[str, object]:
        return self._tensor_dict


class FakeInteractionCache(OrderedDict[str, FakeInteraction]):
    def __init__(self, interactions: list[FakeInteraction]) -> None:
        super().__init__((item.interaction_id, item) for item in interactions)
        self.export_calls: list[tuple[str, float | None]] = []

    def export_interactions(
        self, *, style: str, reward_discount: float | None
    ) -> dict[str, FakeInteraction]:
        self.export_calls.append((style, reward_discount))
        if style == "individual":
            return dict(self)
        if style == "concat":
            parent_ids = {
                item.parent.interaction_id
                for item in self.values()
                if item.parent is not None
            }
            return {
                interaction_id: item
                for interaction_id, item in self.items()
                if interaction_id not in parent_ids
            }
        raise ValueError(style)


def tensor_dict(
    *,
    input_ids: list[int],
    loss_mask: list[int],
    logprobs: list[float],
    versions: list[int],
    reward: float = 1.0,
) -> dict[str, object]:
    return {
        "input_ids": [input_ids],
        "loss_mask": [loss_mask],
        "logprobs": [logprobs],
        "versions": [versions],
        "attention_mask": [[True] * len(input_ids)],
        "rewards": [reward],
    }


class ArealInteractionSidecarTests(unittest.TestCase):
    def _sidecar(self) -> dict[str, object]:
        return build_interaction_adapter_sidecar(
            [
                InteractionBinding(
                    episode_id="episode-1",
                    model_call_id="episode-1:model:tool:0",
                    session_id="session-1",
                    trajectory_id=3,
                    interaction_id="interaction-1",
                    parent_interaction_id=None,
                    ordinal=0,
                    joint_version_id="joint-version-1",
                    route_kind="agent-service-session",
                ),
                InteractionBinding(
                    episode_id="episode-1",
                    model_call_id="episode-1:model:answer:0",
                    session_id="session-1",
                    trajectory_id=3,
                    interaction_id="interaction-2",
                    parent_interaction_id="interaction-1",
                    ordinal=1,
                    joint_version_id="joint-version-1",
                    route_kind="agent-service-session",
                ),
            ]
        )

    def _individual_cache(self) -> FakeInteractionCache:
        first = FakeInteraction(
            interaction_id="interaction-1",
            parent=None,
            chat_template_type="hf",
            input_tokens=[10, 11],
            output_tokens=[20, 21],
            output_logprobs=[-0.2, -0.3],
            output_versions=[7, 7],
            tensor_dict=tensor_dict(
                input_ids=[10, 11, 20, 21],
                loss_mask=[0, 0, 1, 1],
                logprobs=[0.0, 0.0, -0.2, -0.3],
                versions=[-1, -1, 7, 7],
            ),
        )
        second = FakeInteraction(
            interaction_id="interaction-2",
            parent=first,
            chat_template_type="hf",
            input_tokens=[10, 11, 20, 21, 30],
            output_tokens=[40],
            output_logprobs=[-0.4],
            output_versions=[7],
            tensor_dict=tensor_dict(
                input_ids=[10, 11, 20, 21, 30, 40],
                loss_mask=[0, 0, 0, 0, 0, 1],
                logprobs=[0.0, 0.0, 0.0, 0.0, 0.0, -0.4],
                versions=[-1, -1, -1, -1, -1, 7],
            ),
        )
        return FakeInteractionCache([first, second])

    def _concat_cache(self) -> FakeInteractionCache:
        first = FakeInteraction(
            interaction_id="interaction-1",
            parent=None,
            chat_template_type="concat",
            input_tokens=[10, 11],
            output_tokens=[20, 21],
            output_logprobs=[-0.2, -0.3],
            output_versions=[7, 7],
            tensor_dict=tensor_dict(
                input_ids=[10, 11, 20, 21],
                loss_mask=[0, 0, 1, 1],
                logprobs=[0.0, 0.0, -0.2, -0.3],
                versions=[-1, -1, 7, 7],
            ),
        )
        second = FakeInteraction(
            interaction_id="interaction-2",
            parent=first,
            chat_template_type="concat",
            input_tokens=[10, 11, 20, 21, 30],
            output_tokens=[40],
            output_logprobs=[-0.4],
            output_versions=[7],
            tensor_dict=tensor_dict(
                input_ids=[10, 11, 20, 21, 30, 40],
                loss_mask=[0, 0, 1, 1, 0, 1],
                logprobs=[0.0, 0.0, -0.2, -0.3, 0.0, -0.4],
                versions=[-1, -1, 7, 7, -1, 7],
            ),
        )
        return FakeInteractionCache([first, second])

    def test_sidecar_is_one_to_one_and_supports_reverse_lookup(self) -> None:
        sidecar = self._sidecar()
        audit = validate_interaction_adapter_sidecar(sidecar)
        self.assertEqual(audit["binding_count"], 2)
        self.assertEqual(
            interaction_id_for_model_call(
                sidecar, "episode-1:model:answer:0"
            ),
            "interaction-2",
        )
        self.assertEqual(
            model_call_id_for_interaction(sidecar, "interaction-1"),
            "episode-1:model:tool:0",
        )

    def test_sidecar_rejects_duplicate_or_unordered_bindings(self) -> None:
        first = InteractionBinding(
            episode_id="episode-1",
            model_call_id="call-1",
            session_id="session-1",
            trajectory_id=0,
            interaction_id="interaction-1",
            parent_interaction_id=None,
            ordinal=0,
            joint_version_id="joint-1",
            route_kind="agent-service-session",
        )
        duplicate = InteractionBinding(
            episode_id="episode-1",
            model_call_id="call-1",
            session_id="session-1",
            trajectory_id=0,
            interaction_id="interaction-2",
            parent_interaction_id="interaction-1",
            ordinal=1,
            joint_version_id="joint-1",
            route_kind="agent-service-session",
        )
        with self.assertRaisesRegex(
            ArealInteractionAdapterError, "model call ID is bound more than once"
        ):
            build_interaction_adapter_sidecar([first, duplicate])

        child_before_parent = InteractionBinding(
            episode_id="episode-1",
            model_call_id="call-2",
            session_id="session-1",
            trajectory_id=0,
            interaction_id="interaction-2",
            parent_interaction_id="interaction-1",
            ordinal=0,
            joint_version_id="joint-1",
            route_kind="agent-service-session",
        )
        parent_after_child = InteractionBinding(
            episode_id="episode-1",
            model_call_id="call-1",
            session_id="session-1",
            trajectory_id=0,
            interaction_id="interaction-1",
            parent_interaction_id=None,
            ordinal=1,
            joint_version_id="joint-1",
            route_kind="agent-service-session",
        )
        with self.assertRaisesRegex(
            ArealInteractionAdapterError, "parent interaction must precede"
        ):
            build_interaction_adapter_sidecar(
                [child_before_parent, parent_after_child]
            )

    def test_individual_exports_every_call_as_its_own_sample(self) -> None:
        cache = self._individual_cache()
        archive = export_bound_training_sample_archive(
            interaction_cache=cache,
            interaction_sidecar=self._sidecar(),
            export_style="individual",
            turn_discount=1.0,
        )
        audit = validate_bound_training_sample_archive(archive)
        self.assertEqual(audit["sample_count"], 2)
        self.assertEqual(cache.export_calls, [("individual", 1.0)])
        self.assertEqual(
            [sample["included_model_call_ids"] for sample in archive["samples"]],
            [
                ["episode-1:model:tool:0"],
                ["episode-1:model:answer:0"],
            ],
        )
        self.assertEqual(
            archive["samples"][1]["decision_spans"],
            [
                {
                    "model_call_id": "episode-1:model:answer:0",
                    "interaction_id": "interaction-2",
                    "start": 5,
                    "end": 6,
                }
            ],
        )

    def test_concat_exports_leaf_and_preserves_every_decision_span(self) -> None:
        cache = self._concat_cache()
        archive = export_bound_training_sample_archive(
            interaction_cache=cache,
            interaction_sidecar=self._sidecar(),
            export_style="concat",
            turn_discount=1.0,
        )
        audit = validate_bound_training_sample_archive(archive)
        self.assertEqual(audit["sample_count"], 1)
        self.assertEqual(cache.export_calls, [("concat", 1.0)])
        sample = archive["samples"][0]
        self.assertEqual(sample["leaf_interaction_id"], "interaction-2")
        self.assertEqual(
            sample["included_interaction_ids"],
            ["interaction-1", "interaction-2"],
        )
        self.assertEqual(
            sample["included_model_call_ids"],
            ["episode-1:model:tool:0", "episode-1:model:answer:0"],
        )
        self.assertEqual(
            sample["decision_spans"],
            [
                {
                    "model_call_id": "episode-1:model:tool:0",
                    "interaction_id": "interaction-1",
                    "start": 2,
                    "end": 4,
                },
                {
                    "model_call_id": "episode-1:model:answer:0",
                    "interaction_id": "interaction-2",
                    "start": 5,
                    "end": 6,
                },
            ],
        )

    def test_export_fails_closed_on_parent_or_tensor_mismatch(self) -> None:
        cache = self._concat_cache()
        cache["interaction-2"].parent = None
        with self.assertRaisesRegex(
            ArealInteractionAdapterError, "parent relation differs"
        ):
            export_bound_training_sample_archive(
                interaction_cache=cache,
                interaction_sidecar=self._sidecar(),
                export_style="concat",
                turn_discount=1.0,
            )

        cache = self._concat_cache()
        cache["interaction-2"]._tensor_dict["loss_mask"][0][4] = 1
        cache["interaction-2"]._tensor_dict["logprobs"][0][4] = -0.5
        cache["interaction-2"]._tensor_dict["versions"][0][4] = 7
        with self.assertRaisesRegex(
            ArealInteractionAdapterError, "actions not represented"
        ):
            export_bound_training_sample_archive(
                interaction_cache=cache,
                interaction_sidecar=self._sidecar(),
                export_style="concat",
                turn_discount=1.0,
            )

    def test_private_writers_are_root_bounded_and_non_overwriting(self) -> None:
        sidecar = self._sidecar()
        archive = export_bound_training_sample_archive(
            interaction_cache=self._concat_cache(),
            interaction_sidecar=sidecar,
            export_style="concat",
            turn_discount=1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar_path = write_interaction_adapter_sidecar(
                sidecar,
                destination=root / "sidecars" / "episode-1.json",
                allowed_root=root,
            )
            archive_path = write_bound_training_sample_archive(
                archive,
                destination=root / "samples" / "episode-1-concat.json",
                allowed_root=root,
            )
            self.assertEqual(sidecar_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(archive_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                json.loads(archive_path.read_text(encoding="utf-8"))["record_sha256"],
                archive["record_sha256"],
            )
            with self.assertRaises(FileExistsError):
                write_bound_training_sample_archive(
                    archive,
                    destination=archive_path,
                    allowed_root=root,
                )
            with self.assertRaisesRegex(
                ArealInteractionAdapterError, "escapes configured root"
            ):
                write_interaction_adapter_sidecar(
                    sidecar,
                    destination=root.parent / "outside-sidecar.json",
                    allowed_root=root,
                )


if __name__ == "__main__":
    unittest.main()

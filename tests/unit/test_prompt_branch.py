from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sim2real_prompt_annotation.models import (
    PromptPayload,
    RealFrame,
    ReferenceQuery,
)
from sim2real_prompt_annotation.prompt_branch import PromptBranch
from sim2real_prompt_annotation.qwen import QwenOpenAIClient, VLMClient, VLMResponse


def _images() -> tuple[RealFrame, ...]:
    return tuple(
        RealFrame(
            frame_index=index * 10,
            timestamp_seconds=float(index),
            jpeg=f"jpeg-{index}".encode(),
        )
        for index in range(8)
    )


class RecordingClient(VLMClient):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> VLMResponse:
        self.calls.append(kwargs)
        assert kwargs["response_model"] is PromptPayload
        payload = PromptPayload(
            prompt=(
                "A dual-arm robot places a green tennis ball into a storage box "
                "on a white workbench under diffuse overhead lighting."
            ),
            reference_queries=[
                ReferenceQuery(
                    query="green tennis ball", role="primary", required=True
                ),
                ReferenceQuery(query="storage box", role="destination", required=True),
            ],
        )
        return VLMResponse(
            payload=payload,
            raw_text=payload.model_dump_json(),
            model="mock-qwen",
            request_id="request-1",
            input_tokens=100,
            output_tokens=30,
        )


def test_prompt_branch_makes_one_call_for_both_outputs() -> None:
    client = RecordingClient()
    branch = PromptBranch(client, system_prompt="Return the compact JSON response.")

    result = branch.run(
        sample_id="dataset:0",
        task_description="Put the green tennis ball into the storage box",
        robot_metadata={"robot_type": "dual_arm"},
        images=_images(),
    )

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["stage"] == "prompt"
    assert call["images"] == _images()
    assert "Put the green tennis ball into the storage box" in call["user_text"]
    assert '"robot_type":"dual_arm"' in call["user_text"]
    assert result.prompt.startswith("A dual-arm robot places")
    assert [query.query for query in result.reference_queries] == [
        "green tennis ball",
        "storage box",
    ]
    assert result.frame_indices == (0, 10, 20, 30, 40, 50, 60, 70)
    assert result.input_fingerprint.startswith("sha256:")


@pytest.mark.parametrize("count", [0, 1, 7, 9])
def test_prompt_branch_requires_exactly_eight_real_frames(count: int) -> None:
    client = RecordingClient()
    branch = PromptBranch(client, system_prompt="Return JSON.")

    with pytest.raises(ValueError, match="exactly 8 Real frames"):
        branch.run(
            sample_id="dataset:0",
            task_description="move the mug",
            robot_metadata={"robot_type": "dual_arm"},
            images=_images()[:count]
            if count <= 8
            else (
                *_images(),
                RealFrame(frame_index=80, timestamp_seconds=8.0, jpeg=b"jpeg-8"),
            ),
        )

    assert not client.calls


def test_prompt_branch_fingerprint_tracks_visual_input() -> None:
    client = RecordingClient()
    branch = PromptBranch(client, system_prompt="Return JSON.")
    arguments = {
        "sample_id": "dataset:0",
        "task_description": "Put the green tennis ball into the storage box",
        "robot_metadata": {"robot_type": "dual_arm"},
    }

    first = branch.run(**arguments, images=_images())
    changed_images = (
        *_images()[:-1],
        RealFrame(frame_index=70, timestamp_seconds=7.0, jpeg=b"changed-jpeg"),
    )
    second = branch.run(**arguments, images=changed_images)

    assert first.input_fingerprint != second.input_fingerprint


def test_qwen_content_contains_only_ordered_real_jpegs() -> None:
    content = QwenOpenAIClient._content("authoritative metadata", _images())

    assert content[0] == {"type": "text", "text": "authoritative metadata"}
    assert content[1]["text"].startswith("REAL FRAME 1/8:")
    assert content[2]["type"] == "image_url"
    assert content[3]["text"].startswith("REAL FRAME 2/8:")
    image_parts = [part for part in content if part["type"] == "image_url"]
    assert len(image_parts) == 8
    assert all(
        part["image_url"]["url"].startswith("data:image/jpeg;base64,")
        for part in image_parts
    )
    assert not any(part["type"] in {"video", "video_url"} for part in content)
    assert "SIM" not in " ".join(str(part.get("text", "")) for part in content)


def test_qwen_provider_validates_the_compact_payload() -> None:
    payload = PromptPayload(
        prompt="A robot places a mug onto a tray on a white workbench.",
        reference_queries=[
            ReferenceQuery(query="mug", role="primary", required=True),
            ReferenceQuery(query="tray", role="destination", required=True),
        ],
    )
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=payload.model_dump_json()))
        ],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=7),
        model="qwen-test",
        id="request-test",
    )
    create_calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        create_calls.append(kwargs)
        return completion

    client = object.__new__(QwenOpenAIClient)
    client.config = SimpleNamespace(
        model="qwen-test",
        response_format="json_object",
        enable_thinking=False,
    )
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    response = client.generate(
        sample_id="dataset:0",
        stage="prompt",
        system_prompt="Return JSON.",
        user_text="metadata",
        images=_images(),
        response_model=PromptPayload,
        temperature=0.1,
        max_tokens=256,
    )

    assert len(create_calls) == 1
    assert response.payload == payload
    assert response.input_tokens == 12
    assert response.output_tokens == 7
    assert len(create_calls[0]["messages"][1]["content"]) == 17

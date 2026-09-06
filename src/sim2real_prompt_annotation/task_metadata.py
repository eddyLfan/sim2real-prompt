"""Deterministic metadata contracts for prompt-critical task semantics."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import ActiveArm, clean_text


@dataclass(frozen=True)
class TaskContract:
    active_arm: ActiveArm
    active_arm_span: str | None
    task_payload: str


def canonical_robot_description(robot_type: str | None, fallback: str) -> str:
    """Produce a stable robot phrase from authoritative robot metadata."""

    if not robot_type:
        return clean_text(fallback).strip(" .")
    raw = clean_text(robot_type)
    key = re.sub(r"[^a-z0-9]+", "", raw.lower())
    parts: list[str] = []
    if "agilex" in key:
        parts.append("Agilex")
    if "cobotmagic2" in key:
        parts.append("CobotMagic2")
    elif "cobotmagic" in key:
        parts.append("CobotMagic")
    if "dualarm" in key:
        parts.append("dual-arm robot")
    elif "singlearm" in key:
        parts.append("single-arm robot")
    if parts:
        if not any("robot" in part.lower() for part in parts):
            parts.append("robot")
        return " ".join(parts)

    humanized = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw)
    humanized = clean_text(re.sub(r"[_-]+", " ", humanized)).strip(" .")
    return humanized if "robot" in humanized.lower() else f"{humanized} robot"


def task_payload(task: str) -> str:
    """Remove common dataset/robot/date wrappers without interpreting the task."""

    value = clean_text(task)
    if "__" in value:
        value = value.rsplit("__", 1)[1]
    else:
        numbered = list(re.finditer(r"_(?:\d{1,3})_", value))
        for match in reversed(numbered):
            prefix = value[: match.start()].lower()
            if any(
                marker in prefix
                for marker in ("robot", "arm", "gripper", "camera", "cobot")
            ):
                value = value[match.end() :]
                break
    value = re.sub(r"_?\d{8}$", "", value)
    return value.strip(" _-")


def task_contract(task: str | None) -> TaskContract:
    """Extract only deterministic active-arm and payload boundaries."""

    raw = task or ""
    compact = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", raw.lower())
    arm_candidates: tuple[tuple[ActiveArm, tuple[str, ...]], ...] = (
        ("left", ("左臂", "leftarm")),
        ("right", ("右臂", "rightarm")),
        ("both", ("双臂", "usingbotharms", "botharms")),
    )
    active_arm: ActiveArm = "unspecified"
    active_arm_span: str | None = None
    for candidate, markers in arm_candidates:
        marker = next(
            (item for item in markers if item in raw or item in compact), None
        )
        if marker is not None:
            active_arm = candidate
            active_arm_span = marker
            break
    return TaskContract(
        active_arm=active_arm,
        active_arm_span=active_arm_span,
        task_payload=task_payload(raw),
    )

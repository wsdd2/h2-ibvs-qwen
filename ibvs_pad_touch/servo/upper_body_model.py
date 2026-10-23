"""Read-only URDF kinematics for offline waist + arm validation.

No ROS, SDK, publisher, mesh loading, or robot connection. Transform A_T_B maps
coordinates in B into A. Angles passed to this module are radians.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

WAIST_JOINTS = ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint')
RIGHT_ARM_JOINTS = tuple('right_' + n + '_joint' for n in (
    'shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
    'wrist_roll', 'wrist_pitch', 'wrist_yaw'))
POSE_JOINTS = WAIST_JOINTS + RIGHT_ARM_JOINTS


def rotation_axis(axis: Sequence[float], angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all() or np.linalg.norm(a) <= 0:
        raise ValueError('rotation axis must be a finite nonzero 3-vector')
    x, y, z = a / np.linalg.norm(a)
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def transform(xyz=(0., 0., 0.), rpy=(0., 0., 0.)) -> np.ndarray:
    """URDF fixed-axis roll, pitch, yaw: Rz(yaw) Ry(pitch) Rx(roll)."""
    xyz, rpy = np.asarray(xyz, dtype=float), np.asarray(rpy, dtype=float)
    if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite([xyz, rpy]).all():
        raise ValueError('xyz/rpy must be finite 3-vectors')
    out = np.eye(4)
    out[:3, :3] = (rotation_axis((0, 0, 1), rpy[2]) @
                   rotation_axis((0, 1, 0), rpy[1]) @ rotation_axis((1, 0, 0), rpy[0]))
    out[:3, 3] = xyz
    return out


def checked_transform(value) -> np.ndarray:
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all() or
            not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8) or
            not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(t[:3, :3]), 1., atol=1e-6)):
        raise ValueError('expected a finite rigid 4x4 transform')
    return t


def transform_point(a_T_b, point_b) -> np.ndarray:
    t = checked_transform(a_T_b)
    p = np.asarray(point_b, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError('point must be a finite 3-vector')
    return t[:3, :3] @ p + t[:3, 3]


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    velocity: float


class UpperBodyModel:
    def __init__(self, urdf: str | Path):
        self.path = Path(urdf)
        root = ET.parse(self.path).getroot()
        self.links = {e.attrib['name'] for e in root.findall('link')}
        self.joints: dict[str, Joint] = {}
        self.by_child: dict[str, Joint] = {}
        for element in root.findall('joint'):
            name, kind = element.attrib['name'], element.attrib['type']
            if kind not in {'fixed', 'revolute', 'continuous', 'prismatic'}:
                raise ValueError(f'unsupported joint type {kind}: {name}')
            if element.find('mimic') is not None:
                raise ValueError(f'mimic joint not supported: {name}')
            origin = element.find('origin')
            xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()] if origin is not None else [0.] * 3
            rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()] if origin is not None else [0.] * 3
            axis_el, limit = element.find('axis'), element.find('limit')
            axis = np.array([float(v) for v in axis_el.get('xyz', '1 0 0').split()]) if axis_el is not None else np.array([1., 0., 0.])
            if np.linalg.norm(axis) <= 0:
                raise ValueError(f'invalid joint axis: {name}')
            joint = Joint(name, kind, element.find('parent').attrib['link'],
                          element.find('child').attrib['link'], transform(xyz, rpy),
                          axis / np.linalg.norm(axis),
                          float(limit.get('lower', '-inf')) if limit is not None else -math.inf,
                          float(limit.get('upper', 'inf')) if limit is not None else math.inf,
                          float(limit.get('velocity', 'inf')) if limit is not None else math.inf)
            if name in self.joints or joint.child in self.by_child:
                raise ValueError('URDF must be a tree with unique joint names')
            self.joints[name] = joint
            self.by_child[joint.child] = joint
        roots = self.links - self.by_child.keys()
        if len(roots) != 1:
            raise ValueError('URDF must have one root')
        self.root = next(iter(roots))

    def chain(self, link: str) -> list[Joint]:
        if link not in self.links:
            raise ValueError(f'unknown link: {link}')
        out, seen = [], set()
        while link in self.by_child:
            if link in seen:
                raise ValueError('cycle in URDF')
            seen.add(link)
            j = self.by_child[link]
            out.append(j)
            link = j.parent
        return list(reversed(out))

    def _evaluate(self, link: str, q: Mapping[str, float]):
        t = np.eye(4)
        axes, origins = {}, {}
        for j in self.chain(link):
            t = t @ j.origin
            if j.kind != 'fixed':
                if j.name not in q or not math.isfinite(float(q[j.name])):
                    raise ValueError(f'missing/nonfinite joint state: {j.name}')
                origins[j.name] = t[:3, 3].copy()
                axes[j.name] = t[:3, :3] @ j.axis
                motion = np.eye(4)
                if j.kind == 'prismatic':
                    motion[:3, 3] = j.axis * float(q[j.name])
                else:
                    motion[:3, :3] = rotation_axis(j.axis, float(q[j.name]))
                t = t @ motion
        return t, axes, origins

    def root_T(self, link: str, q: Mapping[str, float]) -> np.ndarray:
        return self._evaluate(link, q)[0]

    def relative(self, base: str, tip: str, q: Mapping[str, float]) -> np.ndarray:
        return np.linalg.inv(self.root_T(base, q)) @ self.root_T(tip, q)

    def _world_jacobian(self, link: str, q, names):
        t, axes, origins = self._evaluate(link, q)
        jac = np.zeros((6, len(names)))
        for k, name in enumerate(names):
            if name not in self.joints:
                raise ValueError(f'unknown joint: {name}')
            if name not in axes:
                continue
            if self.joints[name].kind == 'prismatic':
                jac[:3, k] = axes[name]
            else:
                jac[:3, k] = np.cross(axes[name], t[:3, 3] - origins[name])
                jac[3:, k] = axes[name]
        return t, jac

    def jacobian(self, base: str, tip: str, q, names=RIGHT_ARM_JOINTS):
        """Relative geometric Jacobian [linear m/rad; angular rad/rad].

        Subtracts motion of the base as well as rotating axes. In particular,
        waist columns vanish for right_ee_link relative to torso_link.
        """
        a, ja = self._world_jacobian(base, q, names)
        b, jb = self._world_jacobian(tip, q, names)
        out = np.zeros_like(jb)
        delta = b[:3, 3] - a[:3, 3]
        for k in range(len(names)):
            out[:3, k] = a[:3, :3].T @ (jb[:3, k] - ja[:3, k] - np.cross(ja[3:, k], delta))
            out[3:, k] = a[:3, :3].T @ (jb[3:, k] - ja[3:, k])
        return out

    def margins(self, q, names=POSE_JOINTS):
        out = {}
        for n in names:
            v = float(q[n])
            if not math.isfinite(v):
                raise ValueError(f'nonfinite joint: {n}')
            j = self.joints[n]
            out[n] = min(v - j.lower, j.upper - v)
        return out


def arm_singularity_metrics(model: UpperBodyModel, q, length_scale_m=0.30):
    """Dimensionless SVD; arm-only, so waist DOFs cannot hide an arm singularity.

    Scale translation rows by one characteristic arm length. Thresholds are
    offline screening policy, not manufacturer guarantees.
    """
    if not math.isfinite(length_scale_m) or length_scale_m <= 0:
        raise ValueError('length scale must be positive and finite')
    j = model.jacobian('torso_link', 'right_ee_link', q)
    scaled = j.copy()
    scaled[:3] /= length_scale_m
    s = np.linalg.svd(scaled, compute_uv=False)
    pos_s = np.linalg.svd(j[:3], compute_uv=False)
    return {'sigma_min_scaled': float(s[-1]),
            'condition_scaled': float(s[0] / max(s[-1], 1e-15)),
            'sigma_position_min_m_per_rad': float(pos_s[-1]),
            'length_scale_m': float(length_scale_m)}


def camera_target_in_base(model, q, parent_link, parent_T_camera,
                          camera_xyz, base_link='torso_link'):
    """Compose an explicitly supplied calibrated extrinsic with live joints.

    If parent is torso_link and base is torso_link, waist rotation cancels.
    If parent is a wrist, the full wrist chain is included.
    """
    base_T_parent = model.relative(base_link, parent_link, q)
    return transform_point(base_T_parent @ checked_transform(parent_T_camera), camera_xyz)

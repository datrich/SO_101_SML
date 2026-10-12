"""FK / IK so (numpy) cho SO_101_SML, doc truc tiep tu URDF (khong can Isaac).

    kin = SO101Kinematics("description/SO_101_SML.urdf")
    T = kin.fk(q, "gripper_frame_link")          # q: 6 khop (rad), thu tu ARM_JOINTS + gripper
    q = kin.ik(T_target, q0, tcp="grasp")         # damped least squares tren 5 khop canh tay
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINTS = ARM_JOINTS + ["gripper"]


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def tf(R=np.eye(3), p=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


class SO101Kinematics:
    def __init__(self, urdf_path):
        root = ET.parse(urdf_path).getroot()
        self.joints = {}   # child link -> (name, type, parent, T_origin, axis)
        self.limits = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
            ax = j.find("axis")
            axis = [float(v) for v in ax.get("xyz").split()] if ax is not None else [0, 0, 1]
            child = j.find("child").get("link")
            self.joints[child] = (j.get("name"), j.get("type"), j.find("parent").get("link"), tf(rpy_to_R(*rpy), xyz), axis)
            lim = j.find("limit")
            if lim is not None:
                self.limits[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
        self.root = next(iter({p for (_, _, p, _, _) in self.joints.values()} - set(self.joints)))
        self.tcp_offsets = {}  # ten TCP -> (link, T_offset)

    def chain(self, link):
        out = []
        while link != self.root:
            out.append(link)
            link = self.joints[link][2]
        return out[::-1]

    def fk(self, q, link, offset=None):
        qd = dict(zip(JOINTS, q))
        T = np.eye(4)
        for child in self.chain(link):
            name, jtype, _, T0, axis = self.joints[child]
            T = T @ T0
            if jtype in ("revolute", "continuous"):
                T = T @ tf(axis_angle(axis, qd.get(name, 0.0)))
        return T @ offset if offset is not None else T

    def add_tcp(self, name, link, offset):
        self.tcp_offsets[name] = (link, offset)

    def tcp(self, q, name):
        link, off = self.tcp_offsets[name]
        return self.fk(q, link, off)

    def ik(self, T_goal, q0, tcp, iters=300, tol=(1e-4, 1e-3), damping=0.02, w_rot=0.3):
        """DLS tren 5 khop canh tay (gripper giu nguyen). Tra ve (q, loi vi tri m, loi huong rad)."""
        q = np.array(q0, float)
        lo = np.array([self.limits[n][0] for n in ARM_JOINTS])
        hi = np.array([self.limits[n][1] for n in ARM_JOINTS])
        for _ in range(iters):
            T = self.tcp(q, tcp)
            e = self._err(T, T_goal, w_rot)
            if np.linalg.norm(e[:3]) < tol[0] and np.linalg.norm(e[3:]) / w_rot < tol[1]:
                break
            J = np.zeros((6, 5))
            for i in range(5):
                dq = q.copy()
                dq[i] += 1e-6
                J[:, i] = (self._err(self.tcp(dq, tcp), T_goal, w_rot) - e) / -1e-6
            step = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
            q[:5] = np.clip(q[:5] + np.clip(step, -0.2, 0.2), lo, hi)
        e = self._err(self.tcp(q, tcp), T_goal, w_rot)
        return q, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]) / w_rot)

    @staticmethod
    def _err(T, T_goal, w_rot):
        dp = T_goal[:3, 3] - T[:3, 3]
        Re = T_goal[:3, :3] @ T[:3, :3].T
        ang = np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
        if ang < 1e-9:
            w = np.zeros(3)
        else:
            w = ang / (2 * np.sin(ang)) * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
        return np.concatenate([dp, w_rot * w])

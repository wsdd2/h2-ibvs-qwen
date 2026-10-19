from ibvs_pad_touch.web.joints import UPPER_BODY_JOINTS, UpperBodyJointReader

__all__ = ["RealRobotDashboard", "UPPER_BODY_JOINTS", "UpperBodyJointReader"]


def __getattr__(name: str):
    if name == "RealRobotDashboard":
        from ibvs_pad_touch.web.dashboard import RealRobotDashboard

        return RealRobotDashboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

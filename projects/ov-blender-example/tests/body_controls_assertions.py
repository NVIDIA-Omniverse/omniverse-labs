"""Complete numerical body-control contract shared by fixture and shard aggregation."""
import math

CASE_NAMES = ("free", "free_cm", "damped", "box", "box_cm", "capsule", "hinge", "hinge_mass2", "hinge_inertia2", "hinge_principal", "hinge_com", "hinge_density", "hinge_material_density", "hinge_cm", "friction_low", "friction_high", "restitution_low", "restitution_high", "gravity_half", "gravity_x", "static_low", "static_high")
CASE_NAMES += ("hinge_hull", "hinge_hull_cm", "hinge_hull_com", "hinge_hull_com_cm")


def assert_partial_metrics(metrics):
    checks = []
    def check(label, value):
        assert value, label
        checks.append(label)
    # At 0.2s the low-friction acceleration predicts ~0.031m in X; using
    # early motion avoids conflating static friction with later contact sleep.
    if {'static_high', 'static_low'} <= metrics.keys():
        check("static friction retains incline at rest", abs(metrics["static_high"]["position"][0] - .5*math.sin(math.radians(15))) < .001 and .025 < metrics["static_low"]["at_24"][0] - metrics["static_high"]["at_24"][0] < .04)
    if {'hinge_hull_com_cm', 'hinge_hull_com', 'hinge_hull_cm', 'hinge_hull'} <= metrics.keys():
        check("mesh hull COM and inertia meter/cm native equivalence", abs(metrics["hinge_hull_com_cm"]["angle_x"] - metrics["hinge_hull_com"]["angle_x"]) < 1e-4 and abs(metrics["hinge_hull_cm"]["angle_x"] - metrics["hinge_hull"]["angle_x"]) < 1e-4)
    return checks


def assert_metrics(metrics):
    if set(metrics) != set(CASE_NAMES):
        raise AssertionError("Incomplete or unexpected body-controls case roster")
    checks = assert_partial_metrics(metrics)
    def check(label, value):
        assert value, label
        checks.append(label)
    check("linear initial velocity", abs(metrics["free"]["position"][0] - 4)<.03)
    check("angular initial velocity", abs(metrics["free"]["angle_z"] - 2)<.03)
    check("both damping components influence motion", metrics["damped"]["position"][0] < 1.2 and metrics["damped"]["angle_z"] < .6)
    check("box ground contact", abs(metrics["box"]["position"][2] - .5)<.03)
    check("capsule ground contact", abs(metrics["capsule"]["position"][2] - .5)<.03)
    check("linear units equivalence", math.dist(metrics["free"]["position"], metrics["free_cm"]["position"])<1e-4)
    check("box units equivalence", math.dist(metrics["box"]["position"], metrics["box_cm"]["position"])<1e-4)
    base = abs(metrics["hinge"]["angle_x"])
    check("explicit mass changes torque response", abs(metrics["hinge_mass2"]["angle_x"]) > base*1.2)
    check("diagonal inertia changes response", abs(metrics["hinge_inertia2"]["angle_x"]) < base*.8)
    check("principal axes rotate inertia response", abs(metrics["hinge_principal"]["angle_x"]) < base*.8)
    check("COM changes lever response", abs(abs(metrics["hinge_com"]["angle_x"])-base) > .005)
    check("density infers equivalent two-kg mass", abs(metrics["hinge_density"]["angle_x"]-metrics["hinge_mass2"]["angle_x"]) < .005)
    check("material density infers equivalent mass", abs(metrics["hinge_material_density"]["angle_x"]-metrics["hinge_mass2"]["angle_x"]) < .005)
    check("inertia units equivalence", abs(metrics["hinge_cm"]["angle_x"]-metrics["hinge"]["angle_x"])<1e-4)
    check("friction slows sliding", metrics["friction_high"]["position"][0] < metrics["friction_low"]["position"][0]*.4)
    check("restitution causes rebound", metrics["restitution_high"]["rebound_height"] > metrics["restitution_low"]["rebound_height"]+.3)
    check("gravity magnitude changes free fall", metrics["gravity_half"]["at_24"][2] > metrics["restitution_low"]["at_24"][2]+.05)
    check("gravity direction changes free fall", metrics["gravity_x"]["at_24"][0] > .1 and abs(metrics["gravity_x"]["at_24"][2]-2)<.001)

    return checks

from enum import IntEnum

class AxisTypes(IntEnum):
    ZThenX            = 0
    Bisector          = 1
    ZBisect           = 2
    ThreeFold         = 3
    ZOnly             = 4
    NoAxisType        = 5
    LastAxisTypeIndex = 6
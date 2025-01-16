from enum import IntEnum

class YoloType(IntEnum):
    ANCHORS_BASED = 0
    ULTRALYTICS = 1
    SUPERGRAD = 2
    DAMO = 3

    @classmethod
    def from_string(cls, value: str) -> 'YoloType':
        """
        Convert a string representation of the YoloType to its corresponding enum value.

        Args:
            value: The string representation of the YoloType (case-insensitive).

        Returns:
            The corresponding YoloType enum value.

        Raises:
            ValueError: If the string does not match any YoloType.
        """
        # Case-insensitive mapping
        value_upper = value.upper()
        for enum_value in cls:
            if value_upper == enum_value.name.upper():
                return enum_value
        raise ValueError(f"Invalid YoloType string: {value}. Valid options are: {', '.join([e.name for e in cls])}")
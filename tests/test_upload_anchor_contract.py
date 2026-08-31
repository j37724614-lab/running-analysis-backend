from response_chemas import AnchorResultIn


def test_six_point_segmented_calibration_accepts_frontend_distance_fields():
    payload = {
        "points": [
            {"x": 0.90, "y": 0.40},
            {"x": 0.10, "y": 0.40},
            {"x": 0.10, "y": 0.60},
            {"x": 0.90, "y": 0.60},
            {"x": 0.50, "y": 0.40},
            {"x": 0.50, "y": 0.60},
        ],
        "leftToMidDistanceM": 10.0,
        "midToRightDistanceM": 12.0,
    }

    calibration = AnchorResultIn.model_validate(payload)

    assert calibration.leftToMidDistanceM == 10.0
    assert calibration.midToRightDistanceM == 12.0
    assert calibration.segmentedDistanceM == 22.0


def test_six_point_segmented_calibration_builds_a_meter_scale():
    from routes.upload import camera_config_from_stored_calibration

    anchors = [
        {"x": 0.90, "y": 0.40},
        {"x": 0.10, "y": 0.40},
        {"x": 0.10, "y": 0.60},
        {"x": 0.90, "y": 0.60},
        {"x": 0.50, "y": 0.40},
        {"x": 0.50, "y": 0.60},
    ]

    config = camera_config_from_stored_calibration(
        "missing-video.mp4",
        anchors,
        left_to_mid_distance_m=10.0,
        mid_to_right_distance_m=12.0,
    )

    assert config["distance_m"] == 22.0
    assert config["start_line"] == [[0.9, 0.4], [0.9, 0.6]]
    assert config["end_line"] == [[0.1, 0.4], [0.1, 0.6]]

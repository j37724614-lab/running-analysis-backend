def test_missing_average_step_length_remains_unknown():
    from routes.upload import analysis_result_fields

    fields = analysis_result_fields(
        {
            "metrics_csv": None,
            "total_time": 3.2,
            "avg_velocity": None,
            "avg_acceleration": None,
            "avg_step_length": None,
        }
    )

    assert fields["avg_step_length"] is None

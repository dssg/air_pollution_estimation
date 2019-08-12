from traffic_analysis.d04_modelling.tracking.tracking_analyser import TrackingAnalyser


def create_traffic_analysers(params: dict,
                             paths: dict,
                             s3_credentials: dict,
                             verbose=True):
    """Create various traffic analysers with various tracker types
    for evaluation 
    """

    traffic_analysers = {}
    for tracker_type in params["eval_tracker_types"]:
        analyser_name = params["traffic_analyser"].lower()
        traffic_analysers[ f"{analyser_name}_{tracker_type}"] = \
            eval(params["traffic_analyser"])(params=params,
                                             paths=paths,
                                             s3_credentials=s3_credentials,
                                             tracker_type=tracker_type,
                                             verbose=verbose)

    return traffic_analysers
